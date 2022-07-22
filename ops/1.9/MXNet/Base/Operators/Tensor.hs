{-# LANGUAGE AllowAmbiguousTypes, PolyKinds, TypeOperators,
  TypeApplications #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.Base.Operators.Tensor where
import RIO
import RIO.List
import MXNet.Base.Raw
import MXNet.Base.Core.Spec
import MXNet.Base.Core.Enum
import MXNet.Base.Tensor.Class
import MXNet.Base.Types (DType)
import Data.Maybe (catMaybes, fromMaybe)
import Data.Record.Anon.Simple (Record)
import qualified Data.Record.Anon.Simple as Anon

type ParameterList_Activation t u =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("_data", AttrReq (t u))]

_Activation ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_Activation t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_Activation args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Activation t u)) args
        scalarArgs
          = catMaybes
              [("act_type",) . showValue <$> Just (Anon.get #act_type fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Activation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_BatchNorm t u =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

_BatchNorm ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_BatchNorm t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_BatchNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_BatchNorm t u)) args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("moving_mean",) . toRaw <$> Anon.get #moving_mean fullArgs,
               ("moving_var",) . toRaw <$> Anon.get #moving_var fullArgs]
      in
      applyRaw "BatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_BatchNorm_v1 t u =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_BatchNorm_v1 ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_BatchNorm_v1 t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_BatchNorm_v1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_BatchNorm_v1 t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "BatchNorm_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_BilinearSampler t u =
     '[ '("cudnn_off", AttrOpt (Maybe Bool)), '("_data", AttrReq (t u)),
        '("grid", AttrOpt (t u))]

_BilinearSampler ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_BilinearSampler t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_BilinearSampler args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_BilinearSampler t u))
              args
        scalarArgs
          = catMaybes
              [("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("grid",) . toRaw <$> Anon.get #grid fullArgs]
      in
      applyRaw "BilinearSampler" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_BlockGrad t u = '[ '("_data", AttrReq (t u))]

_BlockGrad ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_BlockGrad t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_BlockGrad args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_BlockGrad t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "BlockGrad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_CTCLoss t u =
     '[ '("use_data_lengths", AttrOpt Bool),
        '("use_label_lengths", AttrOpt Bool),
        '("blank_label", AttrOpt (EnumType '["first", "last"])),
        '("_data", AttrReq (t u)), '("label", AttrOpt (t u)),
        '("data_lengths", AttrOpt (t u)),
        '("label_lengths", AttrOpt (t u))]

_CTCLoss ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_CTCLoss t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_CTCLoss args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_CTCLoss t u)) args
        scalarArgs
          = catMaybes
              [("use_data_lengths",) . showValue <$>
                 Anon.get #use_data_lengths fullArgs,
               ("use_label_lengths",) . showValue <$>
                 Anon.get #use_label_lengths fullArgs,
               ("blank_label",) . showValue <$> Anon.get #blank_label fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs,
               ("data_lengths",) . toRaw <$> Anon.get #data_lengths fullArgs,
               ("label_lengths",) . toRaw <$> Anon.get #label_lengths fullArgs]
      in
      applyRaw "CTCLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Cast t u =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"])),
        '("_data", AttrReq (t u))]

_Cast ::
      forall t v u r .
        (Tensor t, FieldsAcc (ParameterList_Cast t u) r, HasCallStack,
         DType v, DType u) =>
        Record r -> TensorApply (t v)
_Cast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Cast t u)) args
        scalarArgs
          = catMaybes
              [("dtype",) . showValue <$> Just (Anon.get #dtype fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Concat t u =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("_data", AttrReq [t u])]

_Concat ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_Concat t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_Concat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Concat t u)) args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("dim",) . showValue <$> Anon.get #dim fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "Concat" scalarArgs (Right tensorVarArgs)

type ParameterList_Convolution t u =
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
        '("_data", AttrReq (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Convolution ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_Convolution t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_Convolution args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Convolution t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("cudnn_tune",) . showValue <$> Anon.get #cudnn_tune fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "Convolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Convolution_v1 t u =
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
        '("_data", AttrReq (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Convolution_v1 ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_Convolution_v1 t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_Convolution_v1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Convolution_v1 t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("cudnn_tune",) . showValue <$> Anon.get #cudnn_tune fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "Convolution_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Correlation t u =
     '[ '("kernel_size", AttrOpt Int),
        '("max_displacement", AttrOpt Int), '("stride1", AttrOpt Int),
        '("stride2", AttrOpt Int), '("pad_size", AttrOpt Int),
        '("is_multiply", AttrOpt Bool), '("data1", AttrOpt (t u)),
        '("data2", AttrOpt (t u))]

_Correlation ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_Correlation t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_Correlation args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Correlation t u))
              args
        scalarArgs
          = catMaybes
              [("kernel_size",) . showValue <$> Anon.get #kernel_size fullArgs,
               ("max_displacement",) . showValue <$>
                 Anon.get #max_displacement fullArgs,
               ("stride1",) . showValue <$> Anon.get #stride1 fullArgs,
               ("stride2",) . showValue <$> Anon.get #stride2 fullArgs,
               ("pad_size",) . showValue <$> Anon.get #pad_size fullArgs,
               ("is_multiply",) . showValue <$> Anon.get #is_multiply fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data1",) . toRaw <$> Anon.get #data1 fullArgs,
               ("data2",) . toRaw <$> Anon.get #data2 fullArgs]
      in
      applyRaw "Correlation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Crop t u =
     '[ '("num_args", AttrReq Int), '("offset", AttrOpt [Int]),
        '("h_w", AttrOpt [Int]), '("center_crop", AttrOpt Bool),
        '("_data", AttrReq [t u])]

_Crop ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_Crop t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_Crop args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Crop t u)) args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("offset",) . showValue <$> Anon.get #offset fullArgs,
               ("h_w",) . showValue <$> Anon.get #h_w fullArgs,
               ("center_crop",) . showValue <$> Anon.get #center_crop fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "Crop" scalarArgs (Right tensorVarArgs)

type ParameterList_CuDNNBatchNorm t u =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

_CuDNNBatchNorm ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_CuDNNBatchNorm t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_CuDNNBatchNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_CuDNNBatchNorm t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("moving_mean",) . toRaw <$> Anon.get #moving_mean fullArgs,
               ("moving_var",) . toRaw <$> Anon.get #moving_var fullArgs]
      in
      applyRaw "CuDNNBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Custom t u =
     '[ '("op_type", AttrOpt Text), '("_data", AttrReq [t u])]

_Custom ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_Custom t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_Custom args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Custom t u)) args
        scalarArgs
          = catMaybes
              [("op_type",) . showValue <$> Anon.get #op_type fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "Custom" scalarArgs (Right tensorVarArgs)

type ParameterList_Deconvolution t u =
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
        '("_data", AttrReq (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Deconvolution ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_Deconvolution t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_Deconvolution args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Deconvolution t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("adj",) . showValue <$> Anon.get #adj fullArgs,
               ("target_shape",) . showValue <$> Anon.get #target_shape fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("cudnn_tune",) . showValue <$> Anon.get #cudnn_tune fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "Deconvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Dropout t u =
     '[ '("p", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["always", "training"])),
        '("axes", AttrOpt [Int]), '("cudnn_off", AttrOpt (Maybe Bool)),
        '("_data", AttrReq (t u))]

_Dropout ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_Dropout t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_Dropout args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Dropout t u)) args
        scalarArgs
          = catMaybes
              [("p",) . showValue <$> Anon.get #p fullArgs,
               ("mode",) . showValue <$> Anon.get #mode fullArgs,
               ("axes",) . showValue <$> Anon.get #axes fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Dropout" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Embedding t u =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("weight", AttrOpt (t u))]

_Embedding ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_Embedding t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_Embedding args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Embedding t u)) args
        scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$>
                 Just (Anon.get #input_dim fullArgs),
               ("output_dim",) . showValue <$>
                 Just (Anon.get #output_dim fullArgs),
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("sparse_grad",) . showValue <$> Anon.get #sparse_grad fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs]
      in
      applyRaw "Embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Flatten t u = '[ '("_data", AttrReq (t u))]

_Flatten ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_Flatten t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_Flatten args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Flatten t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Flatten" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_FullyConnected t u =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u))]

_FullyConnected ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_FullyConnected t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_FullyConnected args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_FullyConnected t u))
              args
        scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 Just (Anon.get #num_hidden fullArgs),
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("flatten",) . showValue <$> Anon.get #flatten fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_GridGenerator t u =
     '[ '("transform_type", AttrReq (EnumType '["affine", "warp"])),
        '("target_shape", AttrOpt [Int]), '("_data", AttrReq (t u))]

_GridGenerator ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_GridGenerator t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_GridGenerator args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_GridGenerator t u))
              args
        scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 Just (Anon.get #transform_type fullArgs),
               ("target_shape",) . showValue <$> Anon.get #target_shape fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "GridGenerator" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_GroupNorm t u =
     '[ '("num_groups", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_GroupNorm ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_GroupNorm t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_GroupNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_GroupNorm t u)) args
        scalarArgs
          = catMaybes
              [("num_groups",) . showValue <$> Anon.get #num_groups fullArgs,
               ("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "GroupNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_IdentityAttachKLSparseReg t u =
     '[ '("sparseness_target", AttrOpt Float),
        '("penalty", AttrOpt Float), '("momentum", AttrOpt Float),
        '("_data", AttrReq (t u))]

_IdentityAttachKLSparseReg ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList_IdentityAttachKLSparseReg t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
_IdentityAttachKLSparseReg args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_IdentityAttachKLSparseReg t u))
              args
        scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 Anon.get #sparseness_target fullArgs,
               ("penalty",) . showValue <$> Anon.get #penalty fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "IdentityAttachKLSparseReg" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_InstanceNorm t u =
     '[ '("eps", AttrOpt Float), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_InstanceNorm ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_InstanceNorm t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_InstanceNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_InstanceNorm t u))
              args
        scalarArgs
          = catMaybes [("eps",) . showValue <$> Anon.get #eps fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "InstanceNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_L2Normalization t u =
     '[ '("eps", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["channel", "instance", "spatial"])),
        '("_data", AttrReq (t u))]

_L2Normalization ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_L2Normalization t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_L2Normalization args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_L2Normalization t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("mode",) . showValue <$> Anon.get #mode fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "L2Normalization" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_LRN t u =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("knorm", AttrOpt Float), '("nsize", AttrReq Int),
        '("_data", AttrReq (t u))]

_LRN ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_LRN t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_LRN args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_LRN t u)) args
        scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs,
               ("knorm",) . showValue <$> Anon.get #knorm fullArgs,
               ("nsize",) . showValue <$> Just (Anon.get #nsize fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "LRN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_LayerNorm t u =
     '[ '("axis", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_LayerNorm ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_LayerNorm t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_LayerNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_LayerNorm t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "LayerNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_LeakyReLU t u =
     '[ '("act_type",
          AttrOpt
            (EnumType '["elu", "gelu", "leaky", "prelu", "rrelu", "selu"])),
        '("slope", AttrOpt Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u))]

_LeakyReLU ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_LeakyReLU t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_LeakyReLU args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_LeakyReLU t u)) args
        scalarArgs
          = catMaybes
              [("act_type",) . showValue <$> Anon.get #act_type fullArgs,
               ("slope",) . showValue <$> Anon.get #slope fullArgs,
               ("lower_bound",) . showValue <$> Anon.get #lower_bound fullArgs,
               ("upper_bound",) . showValue <$> Anon.get #upper_bound fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs]
      in
      applyRaw "LeakyReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_LinearRegressionOutput t u =
     '[ '("grad_scale", AttrOpt Float), '("_data", AttrReq (t u)),
        '("label", AttrOpt (t u))]

_LinearRegressionOutput ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList_LinearRegressionOutput t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
_LinearRegressionOutput args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_LinearRegressionOutput t u))
              args
        scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$> Anon.get #grad_scale fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "LinearRegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_LogisticRegressionOutput t u =
     '[ '("grad_scale", AttrOpt Float), '("_data", AttrReq (t u)),
        '("label", AttrOpt (t u))]

_LogisticRegressionOutput ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList_LogisticRegressionOutput t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
_LogisticRegressionOutput args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_LogisticRegressionOutput t u))
              args
        scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$> Anon.get #grad_scale fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "LogisticRegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_MAERegressionOutput t u =
     '[ '("grad_scale", AttrOpt Float), '("_data", AttrReq (t u)),
        '("label", AttrOpt (t u))]

_MAERegressionOutput ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList_MAERegressionOutput t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
_MAERegressionOutput args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_MAERegressionOutput t u))
              args
        scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$> Anon.get #grad_scale fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "MAERegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_MakeLoss t u =
     '[ '("grad_scale", AttrOpt Float),
        '("valid_thresh", AttrOpt Float),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("_data", AttrReq (t u))]

_MakeLoss ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList_MakeLoss t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
_MakeLoss args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_MakeLoss t u)) args
        scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$> Anon.get #grad_scale fullArgs,
               ("valid_thresh",) . showValue <$> Anon.get #valid_thresh fullArgs,
               ("normalization",) . showValue <$>
                 Anon.get #normalization fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "MakeLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Pad t u =
     '[ '("mode", AttrReq (EnumType '["constant", "edge", "reflect"])),
        '("pad_width", AttrReq [Int]), '("constant_value", AttrOpt Double),
        '("_data", AttrReq (t u))]

_Pad ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_Pad t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_Pad args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Pad t u)) args
        scalarArgs
          = catMaybes
              [("mode",) . showValue <$> Just (Anon.get #mode fullArgs),
               ("pad_width",) . showValue <$> Just (Anon.get #pad_width fullArgs),
               ("constant_value",) . showValue <$>
                 Anon.get #constant_value fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Pooling t u =
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
        '("_data", AttrReq (t u))]

_Pooling ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_Pooling t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_Pooling args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Pooling t u)) args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Anon.get #kernel fullArgs,
               ("pool_type",) . showValue <$> Anon.get #pool_type fullArgs,
               ("global_pool",) . showValue <$> Anon.get #global_pool fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("pooling_convention",) . showValue <$>
                 Anon.get #pooling_convention fullArgs,
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("p_value",) . showValue <$> Anon.get #p_value fullArgs,
               ("count_include_pad",) . showValue <$>
                 Anon.get #count_include_pad fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Pooling_v1 t u =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "max", "sum"])),
        '("global_pool", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("_data", AttrReq (t u))]

_Pooling_v1 ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_Pooling_v1 t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_Pooling_v1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Pooling_v1 t u)) args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Anon.get #kernel fullArgs,
               ("pool_type",) . showValue <$> Anon.get #pool_type fullArgs,
               ("global_pool",) . showValue <$> Anon.get #global_pool fullArgs,
               ("pooling_convention",) . showValue <$>
                 Anon.get #pooling_convention fullArgs,
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Pooling_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_RNN t u =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("projection_size", AttrOpt (Maybe Int)),
        '("lstm_state_clip_min", AttrOpt (Maybe Double)),
        '("lstm_state_clip_max", AttrOpt (Maybe Double)),
        '("lstm_state_clip_nan", AttrOpt Bool),
        '("use_sequence_length", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("parameters", AttrOpt (t u)), '("state", AttrOpt (t u)),
        '("state_cell", AttrOpt (t u)),
        '("sequence_length", AttrOpt (t u))]

_RNN ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_RNN t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_RNN args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_RNN t u)) args
        scalarArgs
          = catMaybes
              [("state_size",) . showValue <$>
                 Just (Anon.get #state_size fullArgs),
               ("num_layers",) . showValue <$>
                 Just (Anon.get #num_layers fullArgs),
               ("bidirectional",) . showValue <$>
                 Anon.get #bidirectional fullArgs,
               ("mode",) . showValue <$> Just (Anon.get #mode fullArgs),
               ("p",) . showValue <$> Anon.get #p fullArgs,
               ("state_outputs",) . showValue <$>
                 Anon.get #state_outputs fullArgs,
               ("projection_size",) . showValue <$>
                 Anon.get #projection_size fullArgs,
               ("lstm_state_clip_min",) . showValue <$>
                 Anon.get #lstm_state_clip_min fullArgs,
               ("lstm_state_clip_max",) . showValue <$>
                 Anon.get #lstm_state_clip_max fullArgs,
               ("lstm_state_clip_nan",) . showValue <$>
                 Anon.get #lstm_state_clip_nan fullArgs,
               ("use_sequence_length",) . showValue <$>
                 Anon.get #use_sequence_length fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("parameters",) . toRaw <$> Anon.get #parameters fullArgs,
               ("state",) . toRaw <$> Anon.get #state fullArgs,
               ("state_cell",) . toRaw <$> Anon.get #state_cell fullArgs,
               ("sequence_length",) . toRaw <$>
                 Anon.get #sequence_length fullArgs]
      in
      applyRaw "RNN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_ROIPooling t u =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("_data", AttrReq (t u)),
        '("rois", AttrOpt (t u))]

_ROIPooling ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_ROIPooling t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_ROIPooling args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ROIPooling t u)) args
        scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 Just (Anon.get #pooled_size fullArgs),
               ("spatial_scale",) . showValue <$>
                 Just (Anon.get #spatial_scale fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("rois",) . toRaw <$> Anon.get #rois fullArgs]
      in
      applyRaw "ROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_Reshape t u =
     '[ '("shape", AttrOpt [Int]), '("reverse", AttrOpt Bool),
        '("target_shape", AttrOpt [Int]), '("keep_highest", AttrOpt Bool),
        '("_data", AttrReq (t u))]

_Reshape ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_Reshape t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_Reshape args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_Reshape t u)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("reverse",) . showValue <$> Anon.get #reverse fullArgs,
               ("target_shape",) . showValue <$> Anon.get #target_shape fullArgs,
               ("keep_highest",) . showValue <$> Anon.get #keep_highest fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "Reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SVMOutput t u =
     '[ '("margin", AttrOpt Float),
        '("regularization_coefficient", AttrOpt Float),
        '("use_linear", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("label", AttrOpt (t u))]

_SVMOutput ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_SVMOutput t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_SVMOutput args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SVMOutput t u)) args
        scalarArgs
          = catMaybes
              [("margin",) . showValue <$> Anon.get #margin fullArgs,
               ("regularization_coefficient",) . showValue <$>
                 Anon.get #regularization_coefficient fullArgs,
               ("use_linear",) . showValue <$> Anon.get #use_linear fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "SVMOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SequenceLast t u =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("_data", AttrReq (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceLast ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_SequenceLast t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_SequenceLast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SequenceLast t u))
              args
        scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 Anon.get #use_sequence_length fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("sequence_length",) . toRaw <$>
                 Anon.get #sequence_length fullArgs]
      in
      applyRaw "SequenceLast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SequenceMask t u =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("value", AttrOpt Float), '("axis", AttrOpt Int),
        '("_data", AttrReq (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceMask ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_SequenceMask t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_SequenceMask args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SequenceMask t u))
              args
        scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 Anon.get #use_sequence_length fullArgs,
               ("value",) . showValue <$> Anon.get #value fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("sequence_length",) . toRaw <$>
                 Anon.get #sequence_length fullArgs]
      in
      applyRaw "SequenceMask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SequenceReverse t u =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("_data", AttrReq (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceReverse ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_SequenceReverse t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_SequenceReverse args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SequenceReverse t u))
              args
        scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 Anon.get #use_sequence_length fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("sequence_length",) . toRaw <$>
                 Anon.get #sequence_length fullArgs]
      in
      applyRaw "SequenceReverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SliceChannel t u =
     '[ '("num_outputs", AttrReq Int), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("_data", AttrReq (t u))]

_SliceChannel ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_SliceChannel t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_SliceChannel args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SliceChannel t u))
              args
        scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 Just (Anon.get #num_outputs fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("squeeze_axis",) . showValue <$> Anon.get #squeeze_axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "SliceChannel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SoftmaxActivation t u =
     '[ '("mode", AttrOpt (EnumType '["channel", "instance"])),
        '("_data", AttrReq (t u))]

_SoftmaxActivation ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_SoftmaxActivation t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_SoftmaxActivation args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_SoftmaxActivation t u))
              args
        scalarArgs
          = catMaybes [("mode",) . showValue <$> Anon.get #mode fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "SoftmaxActivation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SoftmaxOutput t u =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("_data", AttrReq (t u)), '("label", AttrOpt (t u))]

_SoftmaxOutput ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_SoftmaxOutput t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_SoftmaxOutput args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SoftmaxOutput t u))
              args
        scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$> Anon.get #grad_scale fullArgs,
               ("ignore_label",) . showValue <$> Anon.get #ignore_label fullArgs,
               ("multi_output",) . showValue <$> Anon.get #multi_output fullArgs,
               ("use_ignore",) . showValue <$> Anon.get #use_ignore fullArgs,
               ("preserve_shape",) . showValue <$>
                 Anon.get #preserve_shape fullArgs,
               ("normalization",) . showValue <$>
                 Anon.get #normalization fullArgs,
               ("out_grad",) . showValue <$> Anon.get #out_grad fullArgs,
               ("smooth_alpha",) . showValue <$> Anon.get #smooth_alpha fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "SoftmaxOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SpatialTransformer t u =
     '[ '("target_shape", AttrOpt [Int]),
        '("transform_type", AttrReq (EnumType '["affine"])),
        '("sampler_type", AttrReq (EnumType '["bilinear"])),
        '("cudnn_off", AttrOpt (Maybe Bool)), '("_data", AttrReq (t u)),
        '("loc", AttrOpt (t u))]

_SpatialTransformer ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList_SpatialTransformer t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
_SpatialTransformer args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_SpatialTransformer t u))
              args
        scalarArgs
          = catMaybes
              [("target_shape",) . showValue <$> Anon.get #target_shape fullArgs,
               ("transform_type",) . showValue <$>
                 Just (Anon.get #transform_type fullArgs),
               ("sampler_type",) . showValue <$>
                 Just (Anon.get #sampler_type fullArgs),
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("loc",) . toRaw <$> Anon.get #loc fullArgs]
      in
      applyRaw "SpatialTransformer" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_SwapAxis t u =
     '[ '("dim1", AttrOpt Int), '("dim2", AttrOpt Int),
        '("_data", AttrReq (t u))]

_SwapAxis ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList_SwapAxis t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
_SwapAxis args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_SwapAxis t u)) args
        scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> Anon.get #dim1 fullArgs,
               ("dim2",) . showValue <$> Anon.get #dim2 fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "SwapAxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_UpSampling t u =
     '[ '("scale", AttrReq Int), '("num_filter", AttrOpt Int),
        '("sample_type", AttrReq (EnumType '["bilinear", "nearest"])),
        '("multi_input_mode", AttrOpt (EnumType '["concat", "sum"])),
        '("num_args", AttrReq Int), '("workspace", AttrOpt Int),
        '("_data", AttrReq [t u])]

_UpSampling ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_UpSampling t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_UpSampling args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_UpSampling t u)) args
        scalarArgs
          = catMaybes
              [("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("num_filter",) . showValue <$> Anon.get #num_filter fullArgs,
               ("sample_type",) . showValue <$>
                 Just (Anon.get #sample_type fullArgs),
               ("multi_input_mode",) . showValue <$>
                 Anon.get #multi_input_mode fullArgs,
               ("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "UpSampling" scalarArgs (Right tensorVarArgs)

type ParameterList__CachedOp t u = '[ '("_data", AttrReq [t u])]

__CachedOp ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__CachedOp t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__CachedOp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__CachedOp t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_CachedOp" scalarArgs (Right tensorVarArgs)

type ParameterList__CachedOpThreadSafe t u =
     '[ '("_data", AttrReq [t u])]

__CachedOpThreadSafe ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__CachedOpThreadSafe t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__CachedOpThreadSafe args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__CachedOpThreadSafe t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_CachedOpThreadSafe" scalarArgs (Right tensorVarArgs)

type ParameterList__FusedOp t u = '[ '("_data", AttrReq [t u])]

__FusedOp ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__FusedOp t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__FusedOp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__FusedOp t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_FusedOp" scalarArgs (Right tensorVarArgs)

type ParameterList__NoGradient = '[]

__NoGradient ::
             forall t u r .
               (Tensor t, FieldsAcc ParameterList__NoGradient r, HasCallStack,
                DType u) =>
               Record r -> TensorApply (t u)
__NoGradient args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__NoGradient)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_NoGradient" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__adamw_update t u =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("rescale_grad", AttrOpt (t u))]

__adamw_update ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__adamw_update t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__adamw_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__adamw_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("eta",) . showValue <$> Just (Anon.get #eta fullArgs),
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mean",) . toRaw <$> Anon.get #mean fullArgs,
               ("var",) . toRaw <$> Anon.get #var fullArgs,
               ("rescale_grad",) . toRaw <$> Anon.get #rescale_grad fullArgs]
      in
      applyRaw "_adamw_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__arange =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__arange ::
         forall t u r .
           (Tensor t, FieldsAcc ParameterList__arange r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__arange args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__arange)) args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Just (Anon.get #start fullArgs),
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("repeat",) . showValue <$> Anon.get #repeat fullArgs,
               ("infer_range",) . showValue <$> Anon.get #infer_range fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_arange" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Activation = '[]

__backward_Activation ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_Activation r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_Activation args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Activation))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Activation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_BatchNorm = '[]

__backward_BatchNorm ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_BatchNorm r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_BatchNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_BatchNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_BatchNorm_v1 = '[]

__backward_BatchNorm_v1 ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_BatchNorm_v1 r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_BatchNorm_v1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_BatchNorm_v1))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BatchNorm_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_BilinearSampler = '[]

__backward_BilinearSampler ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_BilinearSampler r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_BilinearSampler args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_BilinearSampler))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BilinearSampler" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_CachedOp = '[]

__backward_CachedOp ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_CachedOp r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_CachedOp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_CachedOp))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CachedOp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Concat = '[]

__backward_Concat ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_Concat r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_Concat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Concat))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Concat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Convolution = '[]

__backward_Convolution ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_Convolution r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_Convolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Convolution))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Convolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Convolution_v1 = '[]

__backward_Convolution_v1 ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_Convolution_v1 r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_Convolution_v1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Convolution_v1))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Convolution_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Correlation = '[]

__backward_Correlation ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_Correlation r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_Correlation args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Correlation))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Correlation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Crop = '[]

__backward_Crop ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_Crop r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_Crop args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Crop)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_CuDNNBatchNorm = '[]

__backward_CuDNNBatchNorm ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_CuDNNBatchNorm r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_CuDNNBatchNorm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_CuDNNBatchNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CuDNNBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Custom = '[]

__backward_Custom ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_Custom r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_Custom args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Custom))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Custom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_CustomFunction = '[]

__backward_CustomFunction ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_CustomFunction r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_CustomFunction args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_CustomFunction))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CustomFunction" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Deconvolution = '[]

__backward_Deconvolution ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_Deconvolution r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_Deconvolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Deconvolution))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Deconvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Dropout = '[]

__backward_Dropout ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_Dropout r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_Dropout args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Dropout))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Dropout" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Embedding = '[]

__backward_Embedding ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_Embedding r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_Embedding args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Embedding))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_FullyConnected = '[]

__backward_FullyConnected ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_FullyConnected r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_FullyConnected args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_FullyConnected))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_GridGenerator = '[]

__backward_GridGenerator ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_GridGenerator r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_GridGenerator args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_GridGenerator))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_GridGenerator" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_GroupNorm = '[]

__backward_GroupNorm ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_GroupNorm r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_GroupNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_GroupNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_GroupNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_IdentityAttachKLSparseReg = '[]

__backward_IdentityAttachKLSparseReg ::
                                     forall t u r .
                                       (Tensor t,
                                        FieldsAcc ParameterList__backward_IdentityAttachKLSparseReg
                                          r,
                                        HasCallStack, DType u) =>
                                       Record r -> TensorApply (t u)
__backward_IdentityAttachKLSparseReg args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_IdentityAttachKLSparseReg))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_IdentityAttachKLSparseReg" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_InstanceNorm = '[]

__backward_InstanceNorm ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_InstanceNorm r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_InstanceNorm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_InstanceNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_InstanceNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_L2Normalization = '[]

__backward_L2Normalization ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_L2Normalization r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_L2Normalization args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_L2Normalization))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_L2Normalization" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_LRN = '[]

__backward_LRN ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_LRN r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_LRN args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_LRN)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LRN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_LayerNorm = '[]

__backward_LayerNorm ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_LayerNorm r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_LayerNorm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_LayerNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LayerNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_LeakyReLU = '[]

__backward_LeakyReLU ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_LeakyReLU r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_LeakyReLU args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_LeakyReLU))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LeakyReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_MakeLoss = '[]

__backward_MakeLoss ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_MakeLoss r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_MakeLoss args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_MakeLoss))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_MakeLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Pad = '[]

__backward_Pad ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_Pad r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_Pad args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Pad)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Pooling = '[]

__backward_Pooling ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_Pooling r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_Pooling args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_Pooling))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_Pooling_v1 = '[]

__backward_Pooling_v1 ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_Pooling_v1 r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_Pooling_v1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_Pooling_v1))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pooling_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_RNN = '[]

__backward_RNN ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_RNN r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_RNN args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_RNN)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_RNN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_ROIAlign = '[]

__backward_ROIAlign ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_ROIAlign r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_ROIAlign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_ROIAlign))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_ROIPooling = '[]

__backward_ROIPooling ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_ROIPooling r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_ROIPooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_ROIPooling))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_RROIAlign = '[]

__backward_RROIAlign ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_RROIAlign r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_RROIAlign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_RROIAlign))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_RROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SVMOutput = '[]

__backward_SVMOutput ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_SVMOutput r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_SVMOutput args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_SVMOutput))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SVMOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SequenceLast = '[]

__backward_SequenceLast ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_SequenceLast r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_SequenceLast args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SequenceLast))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceLast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SequenceMask = '[]

__backward_SequenceMask ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_SequenceMask r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_SequenceMask args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SequenceMask))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceMask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SequenceReverse = '[]

__backward_SequenceReverse ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_SequenceReverse r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_SequenceReverse args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SequenceReverse))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceReverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SliceChannel = '[]

__backward_SliceChannel ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_SliceChannel r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_SliceChannel args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SliceChannel))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SliceChannel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SoftmaxActivation = '[]

__backward_SoftmaxActivation ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_SoftmaxActivation r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_SoftmaxActivation args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SoftmaxActivation))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SoftmaxActivation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SoftmaxOutput = '[]

__backward_SoftmaxOutput ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_SoftmaxOutput r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_SoftmaxOutput args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SoftmaxOutput))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SoftmaxOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SparseEmbedding = '[]

__backward_SparseEmbedding ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_SparseEmbedding r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_SparseEmbedding args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SparseEmbedding))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SparseEmbedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SpatialTransformer = '[]

__backward_SpatialTransformer ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__backward_SpatialTransformer r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_SpatialTransformer args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_SpatialTransformer))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SpatialTransformer" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_SwapAxis = '[]

__backward_SwapAxis ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_SwapAxis r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_SwapAxis args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_SwapAxis))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SwapAxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_UpSampling = '[]

__backward_UpSampling ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_UpSampling r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_UpSampling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_UpSampling))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_UpSampling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__CrossDeviceCopy = '[]

__backward__CrossDeviceCopy ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward__CrossDeviceCopy r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward__CrossDeviceCopy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__CrossDeviceCopy))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__CrossDeviceCopy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__NDArray = '[]

__backward__NDArray ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward__NDArray r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward__NDArray args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward__NDArray))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__NDArray" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__Native = '[]

__backward__Native ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward__Native r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward__Native args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward__Native))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__Native" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_DeformableConvolution = '[]

__backward__contrib_DeformableConvolution ::
                                          forall t u r .
                                            (Tensor t,
                                             FieldsAcc
                                               ParameterList__backward__contrib_DeformableConvolution
                                               r,
                                             HasCallStack, DType u) =>
                                            Record r -> TensorApply (t u)
__backward__contrib_DeformableConvolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_DeformableConvolution))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_DeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_DeformablePSROIPooling = '[]

__backward__contrib_DeformablePSROIPooling ::
                                           forall t u r .
                                             (Tensor t,
                                              FieldsAcc
                                                ParameterList__backward__contrib_DeformablePSROIPooling
                                                r,
                                              HasCallStack, DType u) =>
                                             Record r -> TensorApply (t u)
__backward__contrib_DeformablePSROIPooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_DeformablePSROIPooling))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_DeformablePSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_ModulatedDeformableConvolution
     = '[]

__backward__contrib_ModulatedDeformableConvolution ::
                                                   forall t u r .
                                                     (Tensor t,
                                                      FieldsAcc
                                                        ParameterList__backward__contrib_ModulatedDeformableConvolution
                                                        r,
                                                      HasCallStack, DType u) =>
                                                     Record r -> TensorApply (t u)
__backward__contrib_ModulatedDeformableConvolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__backward__contrib_ModulatedDeformableConvolution))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_ModulatedDeformableConvolution"
        scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_MultiBoxDetection = '[]

__backward__contrib_MultiBoxDetection ::
                                      forall t u r .
                                        (Tensor t,
                                         FieldsAcc
                                           ParameterList__backward__contrib_MultiBoxDetection
                                           r,
                                         HasCallStack, DType u) =>
                                        Record r -> TensorApply (t u)
__backward__contrib_MultiBoxDetection args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_MultiBoxDetection))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxDetection" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_MultiBoxPrior = '[]

__backward__contrib_MultiBoxPrior ::
                                  forall t u r .
                                    (Tensor t,
                                     FieldsAcc ParameterList__backward__contrib_MultiBoxPrior r,
                                     HasCallStack, DType u) =>
                                    Record r -> TensorApply (t u)
__backward__contrib_MultiBoxPrior args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_MultiBoxPrior))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxPrior" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_MultiBoxTarget = '[]

__backward__contrib_MultiBoxTarget ::
                                   forall t u r .
                                     (Tensor t,
                                      FieldsAcc ParameterList__backward__contrib_MultiBoxTarget r,
                                      HasCallStack, DType u) =>
                                     Record r -> TensorApply (t u)
__backward__contrib_MultiBoxTarget args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_MultiBoxTarget))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxTarget" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_MultiProposal = '[]

__backward__contrib_MultiProposal ::
                                  forall t u r .
                                    (Tensor t,
                                     FieldsAcc ParameterList__backward__contrib_MultiProposal r,
                                     HasCallStack, DType u) =>
                                    Record r -> TensorApply (t u)
__backward__contrib_MultiProposal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_MultiProposal))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiProposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_PSROIPooling = '[]

__backward__contrib_PSROIPooling ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc ParameterList__backward__contrib_PSROIPooling r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__backward__contrib_PSROIPooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_PSROIPooling))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_PSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_Proposal = '[]

__backward__contrib_Proposal ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward__contrib_Proposal r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward__contrib_Proposal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_Proposal))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_Proposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_SyncBatchNorm = '[]

__backward__contrib_SyncBatchNorm ::
                                  forall t u r .
                                    (Tensor t,
                                     FieldsAcc ParameterList__backward__contrib_SyncBatchNorm r,
                                     HasCallStack, DType u) =>
                                    Record r -> TensorApply (t u)
__backward__contrib_SyncBatchNorm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_SyncBatchNorm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_SyncBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_count_sketch = '[]

__backward__contrib_count_sketch ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc ParameterList__backward__contrib_count_sketch r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__backward__contrib_count_sketch args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_count_sketch))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_count_sketch" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_fft = '[]

__backward__contrib_fft ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward__contrib_fft r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward__contrib_fft args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_fft))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_fft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward__contrib_ifft = '[]

__backward__contrib_ifft ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward__contrib_ifft r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward__contrib_ifft args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward__contrib_ifft))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_ifft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_abs t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_abs ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_abs t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_abs args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_abs t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_abs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_add = '[]

__backward_add ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_add r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_add args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_add)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_amp_cast = '[]

__backward_amp_cast ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_amp_cast r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_amp_cast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_amp_cast))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_amp_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_amp_multicast t u =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("grad", AttrReq [t u])]

__backward_amp_multicast ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__backward_amp_multicast t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_amp_multicast args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_amp_multicast t u))
              args
        scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 Just (Anon.get #num_outputs fullArgs),
               ("cast_narrow",) . showValue <$> Anon.get #cast_narrow fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #grad fullArgs :: [RawTensor t]
      in
      applyRaw "_backward_amp_multicast" scalarArgs (Right tensorVarArgs)

type ParameterList__backward_arccos t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arccos ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__backward_arccos t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_arccos args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arccos t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_arccosh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arccosh ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_arccosh t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_arccosh args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arccosh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_arcsin t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arcsin ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__backward_arcsin t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_arcsin args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arcsin t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_arcsinh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arcsinh ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_arcsinh t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_arcsinh args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arcsinh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_arctan t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arctan ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__backward_arctan t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_arctan args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arctan t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_arctanh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arctanh ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_arctanh t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_arctanh args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_arctanh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_backward_FullyConnected = '[]

__backward_backward_FullyConnected ::
                                   forall t u r .
                                     (Tensor t,
                                      FieldsAcc ParameterList__backward_backward_FullyConnected r,
                                      HasCallStack, DType u) =>
                                     Record r -> TensorApply (t u)
__backward_backward_FullyConnected args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_backward_FullyConnected))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_backward_FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_add = '[]

__backward_broadcast_add ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_broadcast_add r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_broadcast_add args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_add))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_div = '[]

__backward_broadcast_div ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_broadcast_div r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_broadcast_div args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_div))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_exponential =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_exponential ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc ParameterList__backward_broadcast_exponential r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__backward_broadcast_exponential args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_exponential))
              args
        scalarArgs
          = catMaybes
              [("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_gumbel =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_gumbel ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_broadcast_gumbel r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_broadcast_gumbel args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_gumbel))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_gumbel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_hypot = '[]

__backward_broadcast_hypot ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_broadcast_hypot r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_broadcast_hypot args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_hypot))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_logistic =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_logistic ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__backward_broadcast_logistic r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_broadcast_logistic args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_logistic))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_logistic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_maximum = '[]

__backward_broadcast_maximum ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_broadcast_maximum r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_broadcast_maximum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_maximum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_minimum = '[]

__backward_broadcast_minimum ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_broadcast_minimum r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_broadcast_minimum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_minimum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_mod = '[]

__backward_broadcast_mod ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_broadcast_mod r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_broadcast_mod args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_mod))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_mul = '[]

__backward_broadcast_mul ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_broadcast_mul r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_broadcast_mul args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_mul))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_normal =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"]))]

__backward_broadcast_normal ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_broadcast_normal r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_broadcast_normal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_normal))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_pareto =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text)]

__backward_broadcast_pareto ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_broadcast_pareto r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_broadcast_pareto args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_pareto))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_pareto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_power = '[]

__backward_broadcast_power ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_broadcast_power r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_broadcast_power args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_power))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_rayleigh =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_rayleigh ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__backward_broadcast_rayleigh r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_broadcast_rayleigh args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_rayleigh))
              args
        scalarArgs
          = catMaybes
              [("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_rayleigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_sub = '[]

__backward_broadcast_sub ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_broadcast_sub r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_broadcast_sub args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_sub))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_broadcast_weibull =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text)]

__backward_broadcast_weibull ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_broadcast_weibull r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_broadcast_weibull args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_broadcast_weibull))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_weibull" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_cast = '[]

__backward_cast ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_cast r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_cast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_cast)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_cbrt t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cbrt ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_cbrt t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_cbrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_cbrt t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_clip = '[]

__backward_clip ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_clip r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_clip args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_clip)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_clip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_col2im = '[]

__backward_col2im ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_col2im r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_col2im args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_col2im))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_col2im" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_cond = '[]

__backward_cond ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_cond r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_cond args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_cond)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_cond" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_AdaptiveAvgPooling2D = '[]

__backward_contrib_AdaptiveAvgPooling2D ::
                                        forall t u r .
                                          (Tensor t,
                                           FieldsAcc
                                             ParameterList__backward_contrib_AdaptiveAvgPooling2D
                                             r,
                                           HasCallStack, DType u) =>
                                          Record r -> TensorApply (t u)
__backward_contrib_AdaptiveAvgPooling2D args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_AdaptiveAvgPooling2D))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_BatchNormWithReLU = '[]

__backward_contrib_BatchNormWithReLU ::
                                     forall t u r .
                                       (Tensor t,
                                        FieldsAcc ParameterList__backward_contrib_BatchNormWithReLU
                                          r,
                                        HasCallStack, DType u) =>
                                       Record r -> TensorApply (t u)
__backward_contrib_BatchNormWithReLU args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_BatchNormWithReLU))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_BatchNormWithReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_BilinearResize2D = '[]

__backward_contrib_BilinearResize2D ::
                                    forall t u r .
                                      (Tensor t,
                                       FieldsAcc ParameterList__backward_contrib_BilinearResize2D r,
                                       HasCallStack, DType u) =>
                                      Record r -> TensorApply (t u)
__backward_contrib_BilinearResize2D args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_BilinearResize2D))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_BilinearResize2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_bipartite_matching =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int)]

__backward_contrib_bipartite_matching ::
                                      forall t u r .
                                        (Tensor t,
                                         FieldsAcc
                                           ParameterList__backward_contrib_bipartite_matching
                                           r,
                                         HasCallStack, DType u) =>
                                        Record r -> TensorApply (t u)
__backward_contrib_bipartite_matching args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_bipartite_matching))
              args
        scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> Anon.get #is_ascend fullArgs,
               ("threshold",) . showValue <$> Just (Anon.get #threshold fullArgs),
               ("topk",) . showValue <$> Anon.get #topk fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_bipartite_matching" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_boolean_mask =
     '[ '("axis", AttrOpt Int)]

__backward_contrib_boolean_mask ::
                                forall t u r .
                                  (Tensor t,
                                   FieldsAcc ParameterList__backward_contrib_boolean_mask r,
                                   HasCallStack, DType u) =>
                                  Record r -> TensorApply (t u)
__backward_contrib_boolean_mask args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_boolean_mask))
              args
        scalarArgs
          = catMaybes [("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_boolean_mask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_box_iou =
     '[ '("format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_iou ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_contrib_box_iou r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_contrib_box_iou args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_box_iou))
              args
        scalarArgs
          = catMaybes [("format",) . showValue <$> Anon.get #format fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_box_iou" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_contrib_box_nms =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_nms ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_contrib_box_nms r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_contrib_box_nms args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_contrib_box_nms))
              args
        scalarArgs
          = catMaybes
              [("overlap_thresh",) . showValue <$>
                 Anon.get #overlap_thresh fullArgs,
               ("valid_thresh",) . showValue <$> Anon.get #valid_thresh fullArgs,
               ("topk",) . showValue <$> Anon.get #topk fullArgs,
               ("coord_start",) . showValue <$> Anon.get #coord_start fullArgs,
               ("score_index",) . showValue <$> Anon.get #score_index fullArgs,
               ("id_index",) . showValue <$> Anon.get #id_index fullArgs,
               ("background_id",) . showValue <$>
                 Anon.get #background_id fullArgs,
               ("force_suppress",) . showValue <$>
                 Anon.get #force_suppress fullArgs,
               ("in_format",) . showValue <$> Anon.get #in_format fullArgs,
               ("out_format",) . showValue <$> Anon.get #out_format fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_box_nms" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_copy = '[]

__backward_copy ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_copy r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_copy args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_copy)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_cos t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cos ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_cos t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_cos args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_cos t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_cosh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cosh ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_cosh t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_cosh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_cosh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_ctc_loss = '[]

__backward_ctc_loss ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_ctc_loss r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_ctc_loss args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_ctc_loss))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ctc_loss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_degrees t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_degrees ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_degrees t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_degrees args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_degrees t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_diag = '[]

__backward_diag ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_diag r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_diag args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_diag)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_div = '[]

__backward_div ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_div r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_div args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_div)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_div_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__backward_div_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__backward_div_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_div_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_div_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_backward_div_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_dot =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"])))]

__backward_dot ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_dot r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_dot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_dot)) args
        scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$> Anon.get #transpose_a fullArgs,
               ("transpose_b",) . showValue <$> Anon.get #transpose_b fullArgs,
               ("forward_stype",) . showValue <$>
                 Anon.get #forward_stype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_erf t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_erf ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_erf t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_erf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_erf t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_erf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_erfinv t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_erfinv ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__backward_erfinv t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_erfinv args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_erfinv t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_erfinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_expm1 t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_expm1 ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_expm1 t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_expm1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_expm1 t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_foreach = '[]

__backward_foreach ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_foreach r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_foreach args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_foreach))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_foreach" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_gamma t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_gamma ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_gamma t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_gamma t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_gammaln t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_gammaln ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_gammaln t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_gammaln args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_gammaln t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_gammaln" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_gather_nd t u =
     '[ '("shape", AttrReq [Int]), '("_data", AttrReq (t u)),
        '("indices", AttrOpt (t u))]

__backward_gather_nd ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__backward_gather_nd t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_gather_nd args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_gather_nd t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Just (Anon.get #shape fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "_backward_gather_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_hard_sigmoid = '[]

__backward_hard_sigmoid ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_hard_sigmoid r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_hard_sigmoid args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_hard_sigmoid))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_hard_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_hypot = '[]

__backward_hypot ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__backward_hypot r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__backward_hypot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_hypot))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_hypot_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_hypot_scalar ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__backward_hypot_scalar t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_hypot_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_hypot_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_hypot_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_im2col = '[]

__backward_im2col ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_im2col r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_im2col args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_im2col))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_im2col" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_image_crop = '[]

__backward_image_crop ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_image_crop r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_image_crop args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_image_crop))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_image_crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_image_normalize = '[]

__backward_image_normalize ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_image_normalize r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_image_normalize args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_image_normalize))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_image_normalize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_interleaved_matmul_encdec_qk = '[]

__backward_interleaved_matmul_encdec_qk ::
                                        forall t u r .
                                          (Tensor t,
                                           FieldsAcc
                                             ParameterList__backward_interleaved_matmul_encdec_qk
                                             r,
                                           HasCallStack, DType u) =>
                                          Record r -> TensorApply (t u)
__backward_interleaved_matmul_encdec_qk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_interleaved_matmul_encdec_qk))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_encdec_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_interleaved_matmul_encdec_valatt = '[]

__backward_interleaved_matmul_encdec_valatt ::
                                            forall t u r .
                                              (Tensor t,
                                               FieldsAcc
                                                 ParameterList__backward_interleaved_matmul_encdec_valatt
                                                 r,
                                               HasCallStack, DType u) =>
                                              Record r -> TensorApply (t u)
__backward_interleaved_matmul_encdec_valatt args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_interleaved_matmul_encdec_valatt))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_encdec_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_interleaved_matmul_selfatt_qk = '[]

__backward_interleaved_matmul_selfatt_qk ::
                                         forall t u r .
                                           (Tensor t,
                                            FieldsAcc
                                              ParameterList__backward_interleaved_matmul_selfatt_qk
                                              r,
                                            HasCallStack, DType u) =>
                                           Record r -> TensorApply (t u)
__backward_interleaved_matmul_selfatt_qk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_interleaved_matmul_selfatt_qk))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_selfatt_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_interleaved_matmul_selfatt_valatt =
     '[]

__backward_interleaved_matmul_selfatt_valatt ::
                                             forall t u r .
                                               (Tensor t,
                                                FieldsAcc
                                                  ParameterList__backward_interleaved_matmul_selfatt_valatt
                                                  r,
                                                HasCallStack, DType u) =>
                                               Record r -> TensorApply (t u)
__backward_interleaved_matmul_selfatt_valatt args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__backward_interleaved_matmul_selfatt_valatt))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_det = '[]

__backward_linalg_det ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_linalg_det r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_linalg_det args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_det))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_det" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_extractdiag = '[]

__backward_linalg_extractdiag ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__backward_linalg_extractdiag r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_linalg_extractdiag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_extractdiag))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_extractdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_extracttrian = '[]

__backward_linalg_extracttrian ::
                               forall t u r .
                                 (Tensor t, FieldsAcc ParameterList__backward_linalg_extracttrian r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__backward_linalg_extracttrian args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_extracttrian))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_extracttrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_gelqf = '[]

__backward_linalg_gelqf ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_linalg_gelqf r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_linalg_gelqf args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_gelqf))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gelqf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_gemm = '[]

__backward_linalg_gemm ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_linalg_gemm r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_linalg_gemm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_gemm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gemm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_gemm2 = '[]

__backward_linalg_gemm2 ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_linalg_gemm2 r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_linalg_gemm2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_gemm2))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gemm2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_inverse = '[]

__backward_linalg_inverse ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_linalg_inverse r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_linalg_inverse args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_inverse))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_inverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_makediag = '[]

__backward_linalg_makediag ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_linalg_makediag r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_linalg_makediag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_makediag))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_makediag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_maketrian = '[]

__backward_linalg_maketrian ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_linalg_maketrian r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_linalg_maketrian args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_maketrian))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_maketrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_potrf = '[]

__backward_linalg_potrf ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_linalg_potrf r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_linalg_potrf args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_potrf))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_potrf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_potri = '[]

__backward_linalg_potri ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_linalg_potri r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_linalg_potri args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_potri))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_potri" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_slogdet = '[]

__backward_linalg_slogdet ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_linalg_slogdet r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_linalg_slogdet args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_slogdet))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_slogdet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_sumlogdiag = '[]

__backward_linalg_sumlogdiag ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_linalg_sumlogdiag r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_linalg_sumlogdiag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_sumlogdiag))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_sumlogdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_syevd = '[]

__backward_linalg_syevd ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_linalg_syevd r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_linalg_syevd args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_syevd))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_syevd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_syrk = '[]

__backward_linalg_syrk ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_linalg_syrk r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_linalg_syrk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_syrk))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_syrk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_trmm = '[]

__backward_linalg_trmm ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_linalg_trmm r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_linalg_trmm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_trmm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_trmm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linalg_trsm = '[]

__backward_linalg_trsm ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_linalg_trsm r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_linalg_trsm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linalg_trsm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_trsm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_linear_reg_out = '[]

__backward_linear_reg_out ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__backward_linear_reg_out r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_linear_reg_out args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_linear_reg_out))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linear_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_log t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_log t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_log args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_log t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_log10 t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log10 ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_log10 t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_log10 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_log10 t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_log1p t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log1p ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_log1p t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_log1p args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_log1p t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_log2 t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log2 ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_log2 t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_log2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_log2 t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_log_softmax t u =
     '[ '("args", AttrReq [t u])]

__backward_log_softmax ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__backward_log_softmax t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_log_softmax args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_log_softmax t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #args fullArgs :: [RawTensor t]
      in
      applyRaw "_backward_log_softmax" scalarArgs (Right tensorVarArgs)

type ParameterList__backward_logistic_reg_out = '[]

__backward_logistic_reg_out ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_logistic_reg_out r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_logistic_reg_out args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_logistic_reg_out))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_logistic_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mae_reg_out = '[]

__backward_mae_reg_out ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_mae_reg_out r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_mae_reg_out args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_mae_reg_out))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mae_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_max = '[]

__backward_max ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_max r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_max args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_max)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_maximum = '[]

__backward_maximum ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_maximum r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_maximum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_maximum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_maximum_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_maximum_scalar ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__backward_maximum_scalar t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_maximum_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_maximum_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_maximum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mean = '[]

__backward_mean ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_mean r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_mean args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_mean)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_min = '[]

__backward_min ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_min r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_min args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_min)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_minimum = '[]

__backward_minimum ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_minimum r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_minimum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_minimum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_minimum_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_minimum_scalar ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__backward_minimum_scalar t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__backward_minimum_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_minimum_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_minimum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mod = '[]

__backward_mod ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_mod r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_mod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_mod)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_mod_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__backward_mod_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_mod_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_mod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_moments = '[]

__backward_moments ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_moments r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_moments args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_moments))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_moments" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mul = '[]

__backward_mul ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_mul r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_mul args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_mul)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_mul_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__backward_mul_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__backward_mul_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_mul_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_mul_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_backward_mul_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_nanprod = '[]

__backward_nanprod ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_nanprod r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_nanprod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_nanprod))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_nanprod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_nansum = '[]

__backward_nansum ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_nansum r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_nansum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_nansum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_nansum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_norm = '[]

__backward_norm ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_norm r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_norm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_norm)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_average = '[]

__backward_np_average ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_np_average r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_np_average args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_average))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_average" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_broadcast_to = '[]

__backward_np_broadcast_to ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_np_broadcast_to r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_np_broadcast_to args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_broadcast_to))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_column_stack = '[]

__backward_np_column_stack ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_np_column_stack r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_np_column_stack args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_column_stack))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_column_stack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_concat = '[]

__backward_np_concat ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_concat r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_concat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_concat))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_concat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_cumsum = '[]

__backward_np_cumsum ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_cumsum r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_cumsum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_cumsum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_cumsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_diag = '[]

__backward_np_diag ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_np_diag r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_np_diag args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_diag))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_diagflat = '[]

__backward_np_diagflat ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_np_diagflat r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_np_diagflat args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_diagflat))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diagflat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_diagonal = '[]

__backward_np_diagonal ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_np_diagonal r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_np_diagonal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_diagonal))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diagonal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_dot = '[]

__backward_np_dot ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_np_dot r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_np_dot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_dot))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_dstack = '[]

__backward_np_dstack ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_dstack r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_dstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_dstack))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_dstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_hstack = '[]

__backward_np_hstack ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_hstack r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_hstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_hstack))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_hstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_matmul = '[]

__backward_np_matmul ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_matmul r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_matmul args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_matmul))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_matmul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_max = '[]

__backward_np_max ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_np_max r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_np_max args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_max))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_mean = '[]

__backward_np_mean ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_np_mean r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_np_mean args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_mean))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_min = '[]

__backward_np_min ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_np_min r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_np_min args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_min))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_prod = '[]

__backward_np_prod ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_np_prod r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_np_prod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_prod))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_sum = '[]

__backward_np_sum ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_np_sum r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_np_sum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_sum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_trace = '[]

__backward_np_trace ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_np_trace r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_np_trace args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_trace))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_trace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_vstack = '[]

__backward_np_vstack ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_np_vstack r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_np_vstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_vstack))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_vstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_where = '[]

__backward_np_where ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_np_where r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_np_where args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_np_where))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_where_lscalar = '[]

__backward_np_where_lscalar ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_np_where_lscalar r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_np_where_lscalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_where_lscalar))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where_lscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_np_where_rscalar = '[]

__backward_np_where_rscalar ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__backward_np_where_rscalar r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_np_where_rscalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_np_where_rscalar))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where_rscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_arctan2 = '[]

__backward_npi_arctan2 ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_npi_arctan2 r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_npi_arctan2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_arctan2))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_arctan2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_arctan2_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_arctan2_scalar ::
                              forall t u r .
                                (Tensor t,
                                 FieldsAcc (ParameterList__backward_npi_arctan2_scalar t u) r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_npi_arctan2_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_arctan2_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_npi_arctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_add = '[]

__backward_npi_broadcast_add ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_add r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_broadcast_add args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_add))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_div = '[]

__backward_npi_broadcast_div ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_div r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_broadcast_div args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_div))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_mod = '[]

__backward_npi_broadcast_mod ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_mod r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_broadcast_mod args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_mod))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_mul = '[]

__backward_npi_broadcast_mul ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_mul r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_broadcast_mul args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_mul))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_power = '[]

__backward_npi_broadcast_power ::
                               forall t u r .
                                 (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_power r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__backward_npi_broadcast_power args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_power))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_broadcast_sub = '[]

__backward_npi_broadcast_sub ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__backward_npi_broadcast_sub r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_broadcast_sub args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_broadcast_sub))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_copysign = '[]

__backward_npi_copysign ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__backward_npi_copysign r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_npi_copysign args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_copysign))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_copysign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_copysign_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__backward_npi_copysign_scalar ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__backward_npi_copysign_scalar t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__backward_npi_copysign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_copysign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_backward_npi_copysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_diff = '[]

__backward_npi_diff ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_npi_diff r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_npi_diff args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_diff))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_diff" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_einsum = '[]

__backward_npi_einsum ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_npi_einsum r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_npi_einsum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_einsum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_einsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_flip = '[]

__backward_npi_flip ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_npi_flip r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_npi_flip args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_flip))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_flip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_hypot = '[]

__backward_npi_hypot ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_npi_hypot r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_npi_hypot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_hypot))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_ldexp = '[]

__backward_npi_ldexp ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_npi_ldexp r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_npi_ldexp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_ldexp))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_ldexp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_ldexp_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_ldexp_scalar ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__backward_npi_ldexp_scalar t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__backward_npi_ldexp_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_ldexp_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_npi_ldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_norm = '[]

__backward_npi_norm ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__backward_npi_norm r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_npi_norm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_norm))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_pad = '[]

__backward_npi_pad ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_npi_pad r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_npi_pad args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_pad))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_rarctan2_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_rarctan2_scalar ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__backward_npi_rarctan2_scalar t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__backward_npi_rarctan2_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_rarctan2_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_npi_rarctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_rcopysign_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__backward_npi_rcopysign_scalar ::
                                forall t u r .
                                  (Tensor t,
                                   FieldsAcc (ParameterList__backward_npi_rcopysign_scalar t u) r,
                                   HasCallStack, DType u) =>
                                  Record r -> TensorApply (t u)
__backward_npi_rcopysign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_rcopysign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_backward_npi_rcopysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_rldexp_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_rldexp_scalar ::
                             forall t u r .
                               (Tensor t,
                                FieldsAcc (ParameterList__backward_npi_rldexp_scalar t u) r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__backward_npi_rldexp_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_rldexp_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_npi_rldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_solve = '[]

__backward_npi_solve ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_npi_solve r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_npi_solve args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_solve))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_solve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_svd = '[]

__backward_npi_svd ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_npi_svd r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_npi_svd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_npi_svd))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_svd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_tensordot = '[]

__backward_npi_tensordot ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_npi_tensordot r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_npi_tensordot args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_tensordot))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensordot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_tensordot_int_axes = '[]

__backward_npi_tensordot_int_axes ::
                                  forall t u r .
                                    (Tensor t,
                                     FieldsAcc ParameterList__backward_npi_tensordot_int_axes r,
                                     HasCallStack, DType u) =>
                                    Record r -> TensorApply (t u)
__backward_npi_tensordot_int_axes args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_tensordot_int_axes))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensordot_int_axes" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_tensorinv = '[]

__backward_npi_tensorinv ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_npi_tensorinv r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_npi_tensorinv args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_tensorinv))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensorinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_npi_tensorsolve = '[]

__backward_npi_tensorsolve ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_npi_tensorsolve r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_npi_tensorsolve args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_npi_tensorsolve))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensorsolve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_dirichlet = '[]

__backward_pdf_dirichlet ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_pdf_dirichlet r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_pdf_dirichlet args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_dirichlet))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_dirichlet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_exponential = '[]

__backward_pdf_exponential ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__backward_pdf_exponential r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__backward_pdf_exponential args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_exponential))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_gamma = '[]

__backward_pdf_gamma ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__backward_pdf_gamma r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_pdf_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_pdf_gamma))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_generalized_negative_binomial =
     '[]

__backward_pdf_generalized_negative_binomial ::
                                             forall t u r .
                                               (Tensor t,
                                                FieldsAcc
                                                  ParameterList__backward_pdf_generalized_negative_binomial
                                                  r,
                                                HasCallStack, DType u) =>
                                               Record r -> TensorApply (t u)
__backward_pdf_generalized_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__backward_pdf_generalized_negative_binomial))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_negative_binomial = '[]

__backward_pdf_negative_binomial ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc ParameterList__backward_pdf_negative_binomial r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__backward_pdf_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_negative_binomial))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_normal = '[]

__backward_pdf_normal ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_pdf_normal r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_pdf_normal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_normal))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_poisson = '[]

__backward_pdf_poisson ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_pdf_poisson r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_pdf_poisson args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_poisson))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pdf_uniform = '[]

__backward_pdf_uniform ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__backward_pdf_uniform r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_pdf_uniform args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_pdf_uniform))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_pick = '[]

__backward_pick ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_pick r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_pick args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_pick)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pick" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_power = '[]

__backward_power ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__backward_power r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__backward_power args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_power))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_power_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_power_scalar ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__backward_power_scalar t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__backward_power_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_power_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_prod = '[]

__backward_prod ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_prod r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_prod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_prod)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_radians t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_radians ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_radians t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_radians args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_radians t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_rcbrt t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rcbrt ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_rcbrt t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_rcbrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_rcbrt t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_rcbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_rdiv_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rdiv_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__backward_rdiv_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_rdiv_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_rdiv_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_rdiv_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_reciprocal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_reciprocal ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__backward_reciprocal t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_reciprocal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_reciprocal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_relu t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_relu ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_relu t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_relu args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_relu t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_repeat = '[]

__backward_repeat ::
                  forall t u r .
                    (Tensor t, FieldsAcc ParameterList__backward_repeat r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_repeat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_repeat))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_repeat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_reshape = '[]

__backward_reshape ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_reshape r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_reshape args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_reshape))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_reverse = '[]

__backward_reverse ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_reverse r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_reverse args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_reverse))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_reverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_rmod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rmod_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__backward_rmod_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__backward_rmod_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_rmod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_rpower_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rpower_scalar ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__backward_rpower_scalar t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_rpower_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_rpower_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_rsqrt t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rsqrt ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__backward_rsqrt t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__backward_rsqrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_rsqrt t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_rsqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sample_multinomial = '[]

__backward_sample_multinomial ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__backward_sample_multinomial r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__backward_sample_multinomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_sample_multinomial))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sample_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sigmoid t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sigmoid ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_sigmoid t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_sigmoid args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_sigmoid t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sign t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sign ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_sign t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_sign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sign t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sin t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sin ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_sin t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_sin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sin t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sinh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sinh ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_sinh t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_sinh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sinh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_slice = '[]

__backward_slice ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__backward_slice r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__backward_slice args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_slice))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_slice_axis = '[]

__backward_slice_axis ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_slice_axis r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_slice_axis args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_slice_axis))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_slice_like = '[]

__backward_slice_like ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_slice_like r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_slice_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_slice_like))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_smooth_l1 t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_smooth_l1 ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__backward_smooth_l1 t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__backward_smooth_l1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_smooth_l1 t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_smooth_l1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_softmax t u =
     '[ '("args", AttrReq [t u])]

__backward_softmax ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_softmax t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_softmax args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_softmax t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #args fullArgs :: [RawTensor t]
      in applyRaw "_backward_softmax" scalarArgs (Right tensorVarArgs)

type ParameterList__backward_softmax_cross_entropy = '[]

__backward_softmax_cross_entropy ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc ParameterList__backward_softmax_cross_entropy r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__backward_softmax_cross_entropy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_softmax_cross_entropy))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_softmax_cross_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_softmin t u =
     '[ '("args", AttrReq [t u])]

__backward_softmin ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__backward_softmin t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_softmin args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_softmin t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #args fullArgs :: [RawTensor t]
      in applyRaw "_backward_softmin" scalarArgs (Right tensorVarArgs)

type ParameterList__backward_softsign t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_softsign ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__backward_softsign t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__backward_softsign args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_softsign t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_softsign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sparse_retain = '[]

__backward_sparse_retain ::
                         forall t u r .
                           (Tensor t, FieldsAcc ParameterList__backward_sparse_retain r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__backward_sparse_retain args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_sparse_retain))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sparse_retain" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sqrt t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sqrt ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_sqrt t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_sqrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sqrt t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_square t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_square ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__backward_square t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__backward_square args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_square t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_square_sum = '[]

__backward_square_sum ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_square_sum r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_square_sum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_square_sum))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_square_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_squeeze = '[]

__backward_squeeze ::
                   forall t u r .
                     (Tensor t, FieldsAcc ParameterList__backward_squeeze r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__backward_squeeze args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_squeeze))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_stack = '[]

__backward_stack ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__backward_stack r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__backward_stack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_stack))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_stack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sub = '[]

__backward_sub ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_sub r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_sub args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sub)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_sum = '[]

__backward_sum ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__backward_sum r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__backward_sum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_sum)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_take = '[]

__backward_take ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_take r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_take args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_take)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_tan t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_tan ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__backward_tan t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__backward_tan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_tan t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_tanh t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_tanh ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__backward_tanh t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__backward_tanh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_tanh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_backward_tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_tile = '[]

__backward_tile ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_tile r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_tile args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_tile)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_tile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_topk = '[]

__backward_topk ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_topk r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_topk args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_topk)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_topk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_tril = '[]

__backward_tril ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__backward_tril r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__backward_tril args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_tril)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_tril" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_where = '[]

__backward_where ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__backward_where r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__backward_where args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__backward_where))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__backward_while_loop = '[]

__backward_while_loop ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__backward_while_loop r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__backward_while_loop args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__backward_while_loop))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_while_loop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__broadcast_backward = '[]

__broadcast_backward ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__broadcast_backward r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__broadcast_backward args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__broadcast_backward))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_broadcast_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_AdaptiveAvgPooling2D t u =
     '[ '("output_size", AttrOpt [Int]), '("_data", AttrReq (t u))]

__contrib_AdaptiveAvgPooling2D ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__contrib_AdaptiveAvgPooling2D t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__contrib_AdaptiveAvgPooling2D args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_AdaptiveAvgPooling2D t u))
              args
        scalarArgs
          = catMaybes
              [("output_size",) . showValue <$> Anon.get #output_size fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_BatchNormWithReLU t u =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

__contrib_BatchNormWithReLU ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_BatchNormWithReLU t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_BatchNormWithReLU args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_BatchNormWithReLU t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("moving_mean",) . toRaw <$> Anon.get #moving_mean fullArgs,
               ("moving_var",) . toRaw <$> Anon.get #moving_var fullArgs]
      in
      applyRaw "_contrib_BatchNormWithReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_BilinearResize2D t u =
     '[ '("height", AttrOpt Int), '("width", AttrOpt Int),
        '("scale_height", AttrOpt (Maybe Float)),
        '("scale_width", AttrOpt (Maybe Float)),
        '("mode",
          AttrOpt
            (EnumType
               '["like", "odd_scale", "size", "to_even_down", "to_even_up",
                 "to_odd_down", "to_odd_up"])),
        '("align_corners", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("like", AttrOpt (t u))]

__contrib_BilinearResize2D ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList__contrib_BilinearResize2D t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__contrib_BilinearResize2D args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_BilinearResize2D t u))
              args
        scalarArgs
          = catMaybes
              [("height",) . showValue <$> Anon.get #height fullArgs,
               ("width",) . showValue <$> Anon.get #width fullArgs,
               ("scale_height",) . showValue <$> Anon.get #scale_height fullArgs,
               ("scale_width",) . showValue <$> Anon.get #scale_width fullArgs,
               ("mode",) . showValue <$> Anon.get #mode fullArgs,
               ("align_corners",) . showValue <$>
                 Anon.get #align_corners fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("like",) . toRaw <$> Anon.get #like fullArgs]
      in
      applyRaw "_contrib_BilinearResize2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_DeformableConvolution t u =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("_data", AttrReq (t u)), '("offset", AttrOpt (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u))]

__contrib_DeformableConvolution ::
                                forall t u r .
                                  (Tensor t,
                                   FieldsAcc (ParameterList__contrib_DeformableConvolution t u) r,
                                   HasCallStack, DType u) =>
                                  Record r -> TensorApply (t u)
__contrib_DeformableConvolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_DeformableConvolution t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("num_deformable_group",) . showValue <$>
                 Anon.get #num_deformable_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("offset",) . toRaw <$> Anon.get #offset fullArgs,
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "_contrib_DeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_DeformablePSROIPooling t u =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("group_size", AttrReq Int), '("pooled_size", AttrReq Int),
        '("part_size", AttrOpt Int), '("sample_per_part", AttrOpt Int),
        '("trans_std", AttrOpt Float), '("no_trans", AttrOpt Bool),
        '("_data", AttrReq (t u)), '("rois", AttrOpt (t u)),
        '("trans", AttrOpt (t u))]

__contrib_DeformablePSROIPooling ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__contrib_DeformablePSROIPooling t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__contrib_DeformablePSROIPooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_DeformablePSROIPooling t u))
              args
        scalarArgs
          = catMaybes
              [("spatial_scale",) . showValue <$>
                 Just (Anon.get #spatial_scale fullArgs),
               ("output_dim",) . showValue <$>
                 Just (Anon.get #output_dim fullArgs),
               ("group_size",) . showValue <$>
                 Just (Anon.get #group_size fullArgs),
               ("pooled_size",) . showValue <$>
                 Just (Anon.get #pooled_size fullArgs),
               ("part_size",) . showValue <$> Anon.get #part_size fullArgs,
               ("sample_per_part",) . showValue <$>
                 Anon.get #sample_per_part fullArgs,
               ("trans_std",) . showValue <$> Anon.get #trans_std fullArgs,
               ("no_trans",) . showValue <$> Anon.get #no_trans fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("rois",) . toRaw <$> Anon.get #rois fullArgs,
               ("trans",) . toRaw <$> Anon.get #trans fullArgs]
      in
      applyRaw "_contrib_DeformablePSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_ModulatedDeformableConvolution t u =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("im2col_step", AttrOpt Int),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("_data", AttrReq (t u)), '("offset", AttrOpt (t u)),
        '("mask", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

__contrib_ModulatedDeformableConvolution ::
                                         forall t u r .
                                           (Tensor t,
                                            FieldsAcc
                                              (ParameterList__contrib_ModulatedDeformableConvolution
                                                 t
                                                 u)
                                              r,
                                            HasCallStack, DType u) =>
                                           Record r -> TensorApply (t u)
__contrib_ModulatedDeformableConvolution args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__contrib_ModulatedDeformableConvolution t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("num_deformable_group",) . showValue <$>
                 Anon.get #num_deformable_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("im2col_step",) . showValue <$> Anon.get #im2col_step fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("offset",) . toRaw <$> Anon.get #offset fullArgs,
               ("mask",) . toRaw <$> Anon.get #mask fullArgs,
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "_contrib_ModulatedDeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_MultiBoxDetection t u =
     '[ '("clip", AttrOpt Bool), '("threshold", AttrOpt Float),
        '("background_id", AttrOpt Int), '("nms_threshold", AttrOpt Float),
        '("force_suppress", AttrOpt Bool), '("variances", AttrOpt [Float]),
        '("nms_topk", AttrOpt Int), '("cls_prob", AttrOpt (t u)),
        '("loc_pred", AttrOpt (t u)), '("anchor", AttrOpt (t u))]

__contrib_MultiBoxDetection ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_MultiBoxDetection t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_MultiBoxDetection args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_MultiBoxDetection t u))
              args
        scalarArgs
          = catMaybes
              [("clip",) . showValue <$> Anon.get #clip fullArgs,
               ("threshold",) . showValue <$> Anon.get #threshold fullArgs,
               ("background_id",) . showValue <$>
                 Anon.get #background_id fullArgs,
               ("nms_threshold",) . showValue <$>
                 Anon.get #nms_threshold fullArgs,
               ("force_suppress",) . showValue <$>
                 Anon.get #force_suppress fullArgs,
               ("variances",) . showValue <$> Anon.get #variances fullArgs,
               ("nms_topk",) . showValue <$> Anon.get #nms_topk fullArgs]
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> Anon.get #cls_prob fullArgs,
               ("loc_pred",) . toRaw <$> Anon.get #loc_pred fullArgs,
               ("anchor",) . toRaw <$> Anon.get #anchor fullArgs]
      in
      applyRaw "_contrib_MultiBoxDetection" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_MultiBoxPrior t u =
     '[ '("sizes", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("clip", AttrOpt Bool), '("steps", AttrOpt [Float]),
        '("offsets", AttrOpt [Float]), '("_data", AttrReq (t u))]

__contrib_MultiBoxPrior ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_MultiBoxPrior t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_MultiBoxPrior args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_MultiBoxPrior t u))
              args
        scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> Anon.get #sizes fullArgs,
               ("ratios",) . showValue <$> Anon.get #ratios fullArgs,
               ("clip",) . showValue <$> Anon.get #clip fullArgs,
               ("steps",) . showValue <$> Anon.get #steps fullArgs,
               ("offsets",) . showValue <$> Anon.get #offsets fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_MultiBoxPrior" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_MultiBoxTarget t u =
     '[ '("overlap_threshold", AttrOpt Float),
        '("ignore_label", AttrOpt Float),
        '("negative_mining_ratio", AttrOpt Float),
        '("negative_mining_thresh", AttrOpt Float),
        '("minimum_negative_samples", AttrOpt Int),
        '("variances", AttrOpt [Float]), '("anchor", AttrOpt (t u)),
        '("label", AttrOpt (t u)), '("cls_pred", AttrOpt (t u))]

__contrib_MultiBoxTarget ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__contrib_MultiBoxTarget t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__contrib_MultiBoxTarget args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_MultiBoxTarget t u))
              args
        scalarArgs
          = catMaybes
              [("overlap_threshold",) . showValue <$>
                 Anon.get #overlap_threshold fullArgs,
               ("ignore_label",) . showValue <$> Anon.get #ignore_label fullArgs,
               ("negative_mining_ratio",) . showValue <$>
                 Anon.get #negative_mining_ratio fullArgs,
               ("negative_mining_thresh",) . showValue <$>
                 Anon.get #negative_mining_thresh fullArgs,
               ("minimum_negative_samples",) . showValue <$>
                 Anon.get #minimum_negative_samples fullArgs,
               ("variances",) . showValue <$> Anon.get #variances fullArgs]
        tensorKeyArgs
          = catMaybes
              [("anchor",) . toRaw <$> Anon.get #anchor fullArgs,
               ("label",) . toRaw <$> Anon.get #label fullArgs,
               ("cls_pred",) . toRaw <$> Anon.get #cls_pred fullArgs]
      in
      applyRaw "_contrib_MultiBoxTarget" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_MultiProposal t u =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt (t u)),
        '("bbox_pred", AttrOpt (t u)), '("im_info", AttrOpt (t u))]

__contrib_MultiProposal ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_MultiProposal t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_MultiProposal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_MultiProposal t u))
              args
        scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 Anon.get #rpn_pre_nms_top_n fullArgs,
               ("rpn_post_nms_top_n",) . showValue <$>
                 Anon.get #rpn_post_nms_top_n fullArgs,
               ("threshold",) . showValue <$> Anon.get #threshold fullArgs,
               ("rpn_min_size",) . showValue <$> Anon.get #rpn_min_size fullArgs,
               ("scales",) . showValue <$> Anon.get #scales fullArgs,
               ("ratios",) . showValue <$> Anon.get #ratios fullArgs,
               ("feature_stride",) . showValue <$>
                 Anon.get #feature_stride fullArgs,
               ("output_score",) . showValue <$> Anon.get #output_score fullArgs,
               ("iou_loss",) . showValue <$> Anon.get #iou_loss fullArgs]
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> Anon.get #cls_prob fullArgs,
               ("bbox_pred",) . toRaw <$> Anon.get #bbox_pred fullArgs,
               ("im_info",) . toRaw <$> Anon.get #im_info fullArgs]
      in
      applyRaw "_contrib_MultiProposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_PSROIPooling t u =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("pooled_size", AttrReq Int), '("group_size", AttrOpt Int),
        '("_data", AttrReq (t u)), '("rois", AttrOpt (t u))]

__contrib_PSROIPooling ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__contrib_PSROIPooling t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__contrib_PSROIPooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_PSROIPooling t u))
              args
        scalarArgs
          = catMaybes
              [("spatial_scale",) . showValue <$>
                 Just (Anon.get #spatial_scale fullArgs),
               ("output_dim",) . showValue <$>
                 Just (Anon.get #output_dim fullArgs),
               ("pooled_size",) . showValue <$>
                 Just (Anon.get #pooled_size fullArgs),
               ("group_size",) . showValue <$> Anon.get #group_size fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("rois",) . toRaw <$> Anon.get #rois fullArgs]
      in
      applyRaw "_contrib_PSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_Proposal t u =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt (t u)),
        '("bbox_pred", AttrOpt (t u)), '("im_info", AttrOpt (t u))]

__contrib_Proposal ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_Proposal t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_Proposal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_Proposal t u))
              args
        scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 Anon.get #rpn_pre_nms_top_n fullArgs,
               ("rpn_post_nms_top_n",) . showValue <$>
                 Anon.get #rpn_post_nms_top_n fullArgs,
               ("threshold",) . showValue <$> Anon.get #threshold fullArgs,
               ("rpn_min_size",) . showValue <$> Anon.get #rpn_min_size fullArgs,
               ("scales",) . showValue <$> Anon.get #scales fullArgs,
               ("ratios",) . showValue <$> Anon.get #ratios fullArgs,
               ("feature_stride",) . showValue <$>
                 Anon.get #feature_stride fullArgs,
               ("output_score",) . showValue <$> Anon.get #output_score fullArgs,
               ("iou_loss",) . showValue <$> Anon.get #iou_loss fullArgs]
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> Anon.get #cls_prob fullArgs,
               ("bbox_pred",) . toRaw <$> Anon.get #bbox_pred fullArgs,
               ("im_info",) . toRaw <$> Anon.get #im_info fullArgs]
      in
      applyRaw "_contrib_Proposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_ROIAlign t u =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("sample_ratio", AttrOpt Int),
        '("position_sensitive", AttrOpt Bool), '("aligned", AttrOpt Bool),
        '("_data", AttrReq (t u)), '("rois", AttrOpt (t u))]

__contrib_ROIAlign ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_ROIAlign t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_ROIAlign args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_ROIAlign t u))
              args
        scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 Just (Anon.get #pooled_size fullArgs),
               ("spatial_scale",) . showValue <$>
                 Just (Anon.get #spatial_scale fullArgs),
               ("sample_ratio",) . showValue <$> Anon.get #sample_ratio fullArgs,
               ("position_sensitive",) . showValue <$>
                 Anon.get #position_sensitive fullArgs,
               ("aligned",) . showValue <$> Anon.get #aligned fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("rois",) . toRaw <$> Anon.get #rois fullArgs]
      in
      applyRaw "_contrib_ROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_RROIAlign t u =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float),
        '("sampling_ratio", AttrOpt Int), '("_data", AttrReq (t u)),
        '("rois", AttrOpt (t u))]

__contrib_RROIAlign ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__contrib_RROIAlign t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__contrib_RROIAlign args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_RROIAlign t u))
              args
        scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 Just (Anon.get #pooled_size fullArgs),
               ("spatial_scale",) . showValue <$>
                 Just (Anon.get #spatial_scale fullArgs),
               ("sampling_ratio",) . showValue <$>
                 Anon.get #sampling_ratio fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("rois",) . toRaw <$> Anon.get #rois fullArgs]
      in
      applyRaw "_contrib_RROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_SparseEmbedding t u =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("weight", AttrOpt (t u))]

__contrib_SparseEmbedding ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__contrib_SparseEmbedding t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__contrib_SparseEmbedding args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_SparseEmbedding t u))
              args
        scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$>
                 Just (Anon.get #input_dim fullArgs),
               ("output_dim",) . showValue <$>
                 Just (Anon.get #output_dim fullArgs),
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("sparse_grad",) . showValue <$> Anon.get #sparse_grad fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs]
      in
      applyRaw "_contrib_SparseEmbedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_SyncBatchNorm t u =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("ndev", AttrOpt Int),
        '("key", AttrReq Text), '("_data", AttrReq (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u)),
        '("moving_mean", AttrOpt (t u)), '("moving_var", AttrOpt (t u))]

__contrib_SyncBatchNorm ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_SyncBatchNorm t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_SyncBatchNorm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_SyncBatchNorm t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs,
               ("ndev",) . showValue <$> Anon.get #ndev fullArgs,
               ("key",) . showValue <$> Just (Anon.get #key fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("moving_mean",) . toRaw <$> Anon.get #moving_mean fullArgs,
               ("moving_var",) . toRaw <$> Anon.get #moving_var fullArgs]
      in
      applyRaw "_contrib_SyncBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_allclose t u =
     '[ '("rtol", AttrOpt Float), '("atol", AttrOpt Float),
        '("equal_nan", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__contrib_allclose ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_allclose t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_allclose args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_allclose t u))
              args
        scalarArgs
          = catMaybes
              [("rtol",) . showValue <$> Anon.get #rtol fullArgs,
               ("atol",) . showValue <$> Anon.get #atol fullArgs,
               ("equal_nan",) . showValue <$> Anon.get #equal_nan fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_contrib_allclose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_arange_like t u =
     '[ '("start", AttrOpt Double), '("step", AttrOpt Double),
        '("repeat", AttrOpt Int), '("ctx", AttrOpt Text),
        '("axis", AttrOpt (Maybe Int)), '("_data", AttrReq (t u))]

__contrib_arange_like ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__contrib_arange_like t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__contrib_arange_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_arange_like t u))
              args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Anon.get #start fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("repeat",) . showValue <$> Anon.get #repeat fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_arange_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_backward_gradientmultiplier t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__contrib_backward_gradientmultiplier ::
                                      forall t u r .
                                        (Tensor t,
                                         FieldsAcc
                                           (ParameterList__contrib_backward_gradientmultiplier t u)
                                           r,
                                         HasCallStack, DType u) =>
                                        Record r -> TensorApply (t u)
__contrib_backward_gradientmultiplier args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_backward_gradientmultiplier t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_backward_gradientmultiplier" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_backward_hawkesll = '[]

__contrib_backward_hawkesll ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__contrib_backward_hawkesll r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_backward_hawkesll args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_backward_hawkesll))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_hawkesll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_backward_index_copy = '[]

__contrib_backward_index_copy ::
                              forall t u r .
                                (Tensor t, FieldsAcc ParameterList__contrib_backward_index_copy r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__contrib_backward_index_copy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_backward_index_copy))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_index_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_backward_quadratic = '[]

__contrib_backward_quadratic ::
                             forall t u r .
                               (Tensor t, FieldsAcc ParameterList__contrib_backward_quadratic r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__contrib_backward_quadratic args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_backward_quadratic))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_quadratic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_bipartite_matching t u =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int), '("_data", AttrReq (t u))]

__contrib_bipartite_matching ::
                             forall t u r .
                               (Tensor t,
                                FieldsAcc (ParameterList__contrib_bipartite_matching t u) r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__contrib_bipartite_matching args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_bipartite_matching t u))
              args
        scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> Anon.get #is_ascend fullArgs,
               ("threshold",) . showValue <$> Just (Anon.get #threshold fullArgs),
               ("topk",) . showValue <$> Anon.get #topk fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_bipartite_matching" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_boolean_mask t u =
     '[ '("axis", AttrOpt Int), '("_data", AttrReq (t u)),
        '("index", AttrOpt (t u))]

__contrib_boolean_mask ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__contrib_boolean_mask t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__contrib_boolean_mask args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_boolean_mask t u))
              args
        scalarArgs
          = catMaybes [("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("index",) . toRaw <$> Anon.get #index fullArgs]
      in
      applyRaw "_contrib_boolean_mask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_box_decode t u =
     '[ '("std0", AttrOpt Float), '("std1", AttrOpt Float),
        '("std2", AttrOpt Float), '("std3", AttrOpt Float),
        '("clip", AttrOpt Float),
        '("format", AttrOpt (EnumType '["center", "corner"])),
        '("_data", AttrReq (t u)), '("anchors", AttrOpt (t u))]

__contrib_box_decode ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__contrib_box_decode t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__contrib_box_decode args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_box_decode t u))
              args
        scalarArgs
          = catMaybes
              [("std0",) . showValue <$> Anon.get #std0 fullArgs,
               ("std1",) . showValue <$> Anon.get #std1 fullArgs,
               ("std2",) . showValue <$> Anon.get #std2 fullArgs,
               ("std3",) . showValue <$> Anon.get #std3 fullArgs,
               ("clip",) . showValue <$> Anon.get #clip fullArgs,
               ("format",) . showValue <$> Anon.get #format fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("anchors",) . toRaw <$> Anon.get #anchors fullArgs]
      in
      applyRaw "_contrib_box_decode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_box_encode t u =
     '[ '("samples", AttrOpt (t u)), '("matches", AttrOpt (t u)),
        '("anchors", AttrOpt (t u)), '("refs", AttrOpt (t u)),
        '("means", AttrOpt (t u)), '("stds", AttrOpt (t u))]

__contrib_box_encode ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__contrib_box_encode t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__contrib_box_encode args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_box_encode t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("samples",) . toRaw <$> Anon.get #samples fullArgs,
               ("matches",) . toRaw <$> Anon.get #matches fullArgs,
               ("anchors",) . toRaw <$> Anon.get #anchors fullArgs,
               ("refs",) . toRaw <$> Anon.get #refs fullArgs,
               ("means",) . toRaw <$> Anon.get #means fullArgs,
               ("stds",) . toRaw <$> Anon.get #stds fullArgs]
      in
      applyRaw "_contrib_box_encode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_box_iou t u =
     '[ '("format", AttrOpt (EnumType '["center", "corner"])),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__contrib_box_iou ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__contrib_box_iou t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__contrib_box_iou args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_box_iou t u))
              args
        scalarArgs
          = catMaybes [("format",) . showValue <$> Anon.get #format fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_contrib_box_iou" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_box_nms t u =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"])),
        '("_data", AttrReq (t u))]

__contrib_box_nms ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__contrib_box_nms t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__contrib_box_nms args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_box_nms t u))
              args
        scalarArgs
          = catMaybes
              [("overlap_thresh",) . showValue <$>
                 Anon.get #overlap_thresh fullArgs,
               ("valid_thresh",) . showValue <$> Anon.get #valid_thresh fullArgs,
               ("topk",) . showValue <$> Anon.get #topk fullArgs,
               ("coord_start",) . showValue <$> Anon.get #coord_start fullArgs,
               ("score_index",) . showValue <$> Anon.get #score_index fullArgs,
               ("id_index",) . showValue <$> Anon.get #id_index fullArgs,
               ("background_id",) . showValue <$>
                 Anon.get #background_id fullArgs,
               ("force_suppress",) . showValue <$>
                 Anon.get #force_suppress fullArgs,
               ("in_format",) . showValue <$> Anon.get #in_format fullArgs,
               ("out_format",) . showValue <$> Anon.get #out_format fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_box_nms" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_calibrate_entropy t u =
     '[ '("num_quantized_bins", AttrOpt Int), '("hist", AttrOpt (t u)),
        '("hist_edges", AttrOpt (t u))]

__contrib_calibrate_entropy ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_calibrate_entropy t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_calibrate_entropy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_calibrate_entropy t u))
              args
        scalarArgs
          = catMaybes
              [("num_quantized_bins",) . showValue <$>
                 Anon.get #num_quantized_bins fullArgs]
        tensorKeyArgs
          = catMaybes
              [("hist",) . toRaw <$> Anon.get #hist fullArgs,
               ("hist_edges",) . toRaw <$> Anon.get #hist_edges fullArgs]
      in
      applyRaw "_contrib_calibrate_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_count_sketch t u =
     '[ '("out_dim", AttrReq Int),
        '("processing_batch_size", AttrOpt Int), '("_data", AttrReq (t u)),
        '("h", AttrOpt (t u)), '("s", AttrOpt (t u))]

__contrib_count_sketch ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__contrib_count_sketch t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__contrib_count_sketch args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_count_sketch t u))
              args
        scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> Just (Anon.get #out_dim fullArgs),
               ("processing_batch_size",) . showValue <$>
                 Anon.get #processing_batch_size fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("h",) . toRaw <$> Anon.get #h fullArgs,
               ("s",) . toRaw <$> Anon.get #s fullArgs]
      in
      applyRaw "_contrib_count_sketch" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_dequantize t u =
     '[ '("out_type", AttrOpt (EnumType '["float32"])),
        '("_data", AttrReq (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_dequantize ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__contrib_dequantize t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__contrib_dequantize args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_dequantize t u))
              args
        scalarArgs
          = catMaybes
              [("out_type",) . showValue <$> Anon.get #out_type fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_range",) . toRaw <$> Anon.get #min_range fullArgs,
               ("max_range",) . toRaw <$> Anon.get #max_range fullArgs]
      in
      applyRaw "_contrib_dequantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_dgl_adjacency t u =
     '[ '("_data", AttrReq (t u))]

__contrib_dgl_adjacency ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_dgl_adjacency t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_dgl_adjacency args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_dgl_adjacency t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_dgl_adjacency" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_div_sqrt_dim t u =
     '[ '("_data", AttrReq (t u))]

__contrib_div_sqrt_dim ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__contrib_div_sqrt_dim t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__contrib_div_sqrt_dim args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_div_sqrt_dim t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_div_sqrt_dim" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_edge_id t u =
     '[ '("_data", AttrReq (t u)), '("u", AttrOpt (t u)),
        '("v", AttrOpt (t u))]

__contrib_edge_id ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__contrib_edge_id t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__contrib_edge_id args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_edge_id t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("u",) . toRaw <$> Anon.get #u fullArgs,
               ("v",) . toRaw <$> Anon.get #v fullArgs]
      in
      applyRaw "_contrib_edge_id" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_fft t u =
     '[ '("compute_size", AttrOpt Int), '("_data", AttrReq (t u))]

__contrib_fft ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__contrib_fft t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__contrib_fft args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__contrib_fft t u))
              args
        scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$> Anon.get #compute_size fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_fft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_getnnz t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("_data", AttrReq (t u))]

__contrib_getnnz ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__contrib_getnnz t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__contrib_getnnz args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__contrib_getnnz t u))
              args
        scalarArgs
          = catMaybes [("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_getnnz" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_gradientmultiplier t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__contrib_gradientmultiplier ::
                             forall t u r .
                               (Tensor t,
                                FieldsAcc (ParameterList__contrib_gradientmultiplier t u) r,
                                HasCallStack, DType u) =>
                               Record r -> TensorApply (t u)
__contrib_gradientmultiplier args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_gradientmultiplier t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_gradientmultiplier" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_group_adagrad_update t u =
     '[ '("lr", AttrReq Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("history", AttrOpt (t u))]

__contrib_group_adagrad_update ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__contrib_group_adagrad_update t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__contrib_group_adagrad_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_group_adagrad_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("history",) . toRaw <$> Anon.get #history fullArgs]
      in
      applyRaw "_contrib_group_adagrad_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_hawkesll t u =
     '[ '("lda", AttrOpt (t u)), '("alpha", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("state", AttrOpt (t u)),
        '("lags", AttrOpt (t u)), '("marks", AttrOpt (t u)),
        '("valid_length", AttrOpt (t u)), '("max_time", AttrOpt (t u))]

__contrib_hawkesll ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_hawkesll t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_hawkesll args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_hawkesll t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lda",) . toRaw <$> Anon.get #lda fullArgs,
               ("alpha",) . toRaw <$> Anon.get #alpha fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("state",) . toRaw <$> Anon.get #state fullArgs,
               ("lags",) . toRaw <$> Anon.get #lags fullArgs,
               ("marks",) . toRaw <$> Anon.get #marks fullArgs,
               ("valid_length",) . toRaw <$> Anon.get #valid_length fullArgs,
               ("max_time",) . toRaw <$> Anon.get #max_time fullArgs]
      in
      applyRaw "_contrib_hawkesll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_ifft t u =
     '[ '("compute_size", AttrOpt Int), '("_data", AttrReq (t u))]

__contrib_ifft ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__contrib_ifft t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__contrib_ifft args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__contrib_ifft t u))
              args
        scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$> Anon.get #compute_size fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_ifft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_index_array t u =
     '[ '("axes", AttrOpt (Maybe [Int])), '("_data", AttrReq (t u))]

__contrib_index_array ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__contrib_index_array t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__contrib_index_array args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_index_array t u))
              args
        scalarArgs
          = catMaybes [("axes",) . showValue <$> Anon.get #axes fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_index_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_index_copy t u =
     '[ '("old_tensor", AttrOpt (t u)),
        '("index_vector", AttrOpt (t u)), '("new_tensor", AttrOpt (t u))]

__contrib_index_copy ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__contrib_index_copy t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__contrib_index_copy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_index_copy t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("old_tensor",) . toRaw <$> Anon.get #old_tensor fullArgs,
               ("index_vector",) . toRaw <$> Anon.get #index_vector fullArgs,
               ("new_tensor",) . toRaw <$> Anon.get #new_tensor fullArgs]
      in
      applyRaw "_contrib_index_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_interleaved_matmul_encdec_qk t u =
     '[ '("heads", AttrReq Int), '("queries", AttrOpt (t u)),
        '("keys_values", AttrOpt (t u))]

__contrib_interleaved_matmul_encdec_qk ::
                                       forall t u r .
                                         (Tensor t,
                                          FieldsAcc
                                            (ParameterList__contrib_interleaved_matmul_encdec_qk t
                                               u)
                                            r,
                                          HasCallStack, DType u) =>
                                         Record r -> TensorApply (t u)
__contrib_interleaved_matmul_encdec_qk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_interleaved_matmul_encdec_qk t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("queries",) . toRaw <$> Anon.get #queries fullArgs,
               ("keys_values",) . toRaw <$> Anon.get #keys_values fullArgs]
      in
      applyRaw "_contrib_interleaved_matmul_encdec_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_interleaved_matmul_encdec_valatt t u =
     '[ '("heads", AttrReq Int), '("keys_values", AttrOpt (t u)),
        '("attention", AttrOpt (t u))]

__contrib_interleaved_matmul_encdec_valatt ::
                                           forall t u r .
                                             (Tensor t,
                                              FieldsAcc
                                                (ParameterList__contrib_interleaved_matmul_encdec_valatt
                                                   t
                                                   u)
                                                r,
                                              HasCallStack, DType u) =>
                                             Record r -> TensorApply (t u)
__contrib_interleaved_matmul_encdec_valatt args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__contrib_interleaved_matmul_encdec_valatt t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("keys_values",) . toRaw <$> Anon.get #keys_values fullArgs,
               ("attention",) . toRaw <$> Anon.get #attention fullArgs]
      in
      applyRaw "_contrib_interleaved_matmul_encdec_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_interleaved_matmul_selfatt_qk t u =
     '[ '("heads", AttrReq Int),
        '("queries_keys_values", AttrOpt (t u))]

__contrib_interleaved_matmul_selfatt_qk ::
                                        forall t u r .
                                          (Tensor t,
                                           FieldsAcc
                                             (ParameterList__contrib_interleaved_matmul_selfatt_qk t
                                                u)
                                             r,
                                           HasCallStack, DType u) =>
                                          Record r -> TensorApply (t u)
__contrib_interleaved_matmul_selfatt_qk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_interleaved_matmul_selfatt_qk t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 Anon.get #queries_keys_values fullArgs]
      in
      applyRaw "_contrib_interleaved_matmul_selfatt_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_interleaved_matmul_selfatt_valatt t u =
     '[ '("heads", AttrReq Int),
        '("queries_keys_values", AttrOpt (t u)),
        '("attention", AttrOpt (t u))]

__contrib_interleaved_matmul_selfatt_valatt ::
                                            forall t u r .
                                              (Tensor t,
                                               FieldsAcc
                                                 (ParameterList__contrib_interleaved_matmul_selfatt_valatt
                                                    t
                                                    u)
                                                 r,
                                               HasCallStack, DType u) =>
                                              Record r -> TensorApply (t u)
__contrib_interleaved_matmul_selfatt_valatt args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__contrib_interleaved_matmul_selfatt_valatt t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 Anon.get #queries_keys_values fullArgs,
               ("attention",) . toRaw <$> Anon.get #attention fullArgs]
      in
      applyRaw "_contrib_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_intgemm_fully_connected t u =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool),
        '("out_type", AttrOpt (EnumType '["float32", "int32"])),
        '("_data", AttrReq (t u)), '("weight", AttrOpt (t u)),
        '("scaling", AttrOpt (t u)), '("bias", AttrOpt (t u))]

__contrib_intgemm_fully_connected ::
                                  forall t u r .
                                    (Tensor t,
                                     FieldsAcc (ParameterList__contrib_intgemm_fully_connected t u)
                                       r,
                                     HasCallStack, DType u) =>
                                    Record r -> TensorApply (t u)
__contrib_intgemm_fully_connected args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_intgemm_fully_connected t u))
              args
        scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 Just (Anon.get #num_hidden fullArgs),
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("flatten",) . showValue <$> Anon.get #flatten fullArgs,
               ("out_type",) . showValue <$> Anon.get #out_type fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("scaling",) . toRaw <$> Anon.get #scaling fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs]
      in
      applyRaw "_contrib_intgemm_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_intgemm_maxabsolute t u =
     '[ '("_data", AttrReq (t u))]

__contrib_intgemm_maxabsolute ::
                              forall t u r .
                                (Tensor t,
                                 FieldsAcc (ParameterList__contrib_intgemm_maxabsolute t u) r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__contrib_intgemm_maxabsolute args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_intgemm_maxabsolute t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_intgemm_maxabsolute" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_intgemm_prepare_data t u =
     '[ '("_data", AttrReq (t u)), '("maxabs", AttrOpt (t u))]

__contrib_intgemm_prepare_data ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__contrib_intgemm_prepare_data t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__contrib_intgemm_prepare_data args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_intgemm_prepare_data t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("maxabs",) . toRaw <$> Anon.get #maxabs fullArgs]
      in
      applyRaw "_contrib_intgemm_prepare_data" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_intgemm_prepare_weight t u =
     '[ '("already_quantized", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("maxabs", AttrOpt (t u))]

__contrib_intgemm_prepare_weight ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__contrib_intgemm_prepare_weight t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__contrib_intgemm_prepare_weight args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_intgemm_prepare_weight t u))
              args
        scalarArgs
          = catMaybes
              [("already_quantized",) . showValue <$>
                 Anon.get #already_quantized fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("maxabs",) . toRaw <$> Anon.get #maxabs fullArgs]
      in
      applyRaw "_contrib_intgemm_prepare_weight" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_intgemm_take_weight t u =
     '[ '("weight", AttrOpt (t u)), '("indices", AttrOpt (t u))]

__contrib_intgemm_take_weight ::
                              forall t u r .
                                (Tensor t,
                                 FieldsAcc (ParameterList__contrib_intgemm_take_weight t u) r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__contrib_intgemm_take_weight args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_intgemm_take_weight t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "_contrib_intgemm_take_weight" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_mrcnn_mask_target t u =
     '[ '("num_rois", AttrReq Int), '("num_classes", AttrReq Int),
        '("mask_size", AttrReq [Int]), '("sample_ratio", AttrOpt Int),
        '("aligned", AttrOpt Bool), '("rois", AttrOpt (t u)),
        '("gt_masks", AttrOpt (t u)), '("matches", AttrOpt (t u)),
        '("cls_targets", AttrOpt (t u))]

__contrib_mrcnn_mask_target ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_mrcnn_mask_target t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_mrcnn_mask_target args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_mrcnn_mask_target t u))
              args
        scalarArgs
          = catMaybes
              [("num_rois",) . showValue <$> Just (Anon.get #num_rois fullArgs),
               ("num_classes",) . showValue <$>
                 Just (Anon.get #num_classes fullArgs),
               ("mask_size",) . showValue <$> Just (Anon.get #mask_size fullArgs),
               ("sample_ratio",) . showValue <$> Anon.get #sample_ratio fullArgs,
               ("aligned",) . showValue <$> Anon.get #aligned fullArgs]
        tensorKeyArgs
          = catMaybes
              [("rois",) . toRaw <$> Anon.get #rois fullArgs,
               ("gt_masks",) . toRaw <$> Anon.get #gt_masks fullArgs,
               ("matches",) . toRaw <$> Anon.get #matches fullArgs,
               ("cls_targets",) . toRaw <$> Anon.get #cls_targets fullArgs]
      in
      applyRaw "_contrib_mrcnn_mask_target" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quadratic t u =
     '[ '("a", AttrOpt Float), '("b", AttrOpt Float),
        '("c", AttrOpt Float), '("_data", AttrReq (t u))]

__contrib_quadratic ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__contrib_quadratic t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__contrib_quadratic args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quadratic t u))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("b",) . showValue <$> Anon.get #b fullArgs,
               ("c",) . showValue <$> Anon.get #c fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_quadratic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantize t u =
     '[ '("out_type", AttrOpt (EnumType '["int8", "uint8"])),
        '("_data", AttrReq (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_quantize ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_quantize t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_quantize args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantize t u))
              args
        scalarArgs
          = catMaybes
              [("out_type",) . showValue <$> Anon.get #out_type fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_range",) . toRaw <$> Anon.get #min_range fullArgs,
               ("max_range",) . toRaw <$> Anon.get #max_range fullArgs]
      in
      applyRaw "_contrib_quantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantize_asym t u =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u))]

__contrib_quantize_asym ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_quantize_asym t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_quantize_asym args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantize_asym t u))
              args
        scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_quantize_asym" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantize_v2 t u =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u))]

__contrib_quantize_v2 ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__contrib_quantize_v2 t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__contrib_quantize_v2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantize_v2 t u))
              args
        scalarArgs
          = catMaybes
              [("out_type",) . showValue <$> Anon.get #out_type fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_quantize_v2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_act t u =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("_data", AttrReq (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_act ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_quantized_act t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_quantized_act args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_act t u))
              args
        scalarArgs
          = catMaybes
              [("act_type",) . showValue <$> Just (Anon.get #act_type fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs]
      in
      applyRaw "_contrib_quantized_act" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_batch_norm t u =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_batch_norm ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__contrib_quantized_batch_norm t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__contrib_quantized_batch_norm args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_batch_norm t u))
              args
        scalarArgs
          = catMaybes
              [("eps",) . showValue <$> Anon.get #eps fullArgs,
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("fix_gamma",) . showValue <$> Anon.get #fix_gamma fullArgs,
               ("use_global_stats",) . showValue <$>
                 Anon.get #use_global_stats fullArgs,
               ("output_mean_var",) . showValue <$>
                 Anon.get #output_mean_var fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("gamma",) . toRaw <$> Anon.get #gamma fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs,
               ("moving_mean",) . toRaw <$> Anon.get #moving_mean fullArgs,
               ("moving_var",) . toRaw <$> Anon.get #moving_var fullArgs,
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs]
      in
      applyRaw "_contrib_quantized_batch_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_concat t u =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("_data", AttrReq [t u])]

__contrib_quantized_concat ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList__contrib_quantized_concat t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__contrib_quantized_concat args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_concat t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("dim",) . showValue <$> Anon.get #dim fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "_contrib_quantized_concat" scalarArgs
        (Right tensorVarArgs)

type ParameterList__contrib_quantized_conv t u =
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
        '("_data", AttrReq (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u)), '("min_weight", AttrOpt (t u)),
        '("max_weight", AttrOpt (t u)), '("min_bias", AttrOpt (t u)),
        '("max_bias", AttrOpt (t u))]

__contrib_quantized_conv ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__contrib_quantized_conv t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__contrib_quantized_conv args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_conv t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("num_filter",) . showValue <$>
                 Just (Anon.get #num_filter fullArgs),
               ("num_group",) . showValue <$> Anon.get #num_group fullArgs,
               ("workspace",) . showValue <$> Anon.get #workspace fullArgs,
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("cudnn_tune",) . showValue <$> Anon.get #cudnn_tune fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs,
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs,
               ("min_weight",) . toRaw <$> Anon.get #min_weight fullArgs,
               ("max_weight",) . toRaw <$> Anon.get #max_weight fullArgs,
               ("min_bias",) . toRaw <$> Anon.get #min_bias fullArgs,
               ("max_bias",) . toRaw <$> Anon.get #max_bias fullArgs]
      in
      applyRaw "_contrib_quantized_conv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_elemwise_add t u =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u)),
        '("lhs_min", AttrOpt (t u)), '("lhs_max", AttrOpt (t u)),
        '("rhs_min", AttrOpt (t u)), '("rhs_max", AttrOpt (t u))]

__contrib_quantized_elemwise_add ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__contrib_quantized_elemwise_add t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__contrib_quantized_elemwise_add args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_elemwise_add t u))
              args
        scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs,
               ("lhs_min",) . toRaw <$> Anon.get #lhs_min fullArgs,
               ("lhs_max",) . toRaw <$> Anon.get #lhs_max fullArgs,
               ("rhs_min",) . toRaw <$> Anon.get #rhs_min fullArgs,
               ("rhs_max",) . toRaw <$> Anon.get #rhs_max fullArgs]
      in
      applyRaw "_contrib_quantized_elemwise_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_elemwise_mul t u =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("enable_float_output", AttrOpt Bool), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u)), '("lhs_min", AttrOpt (t u)),
        '("lhs_max", AttrOpt (t u)), '("rhs_min", AttrOpt (t u)),
        '("rhs_max", AttrOpt (t u))]

__contrib_quantized_elemwise_mul ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__contrib_quantized_elemwise_mul t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__contrib_quantized_elemwise_mul args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_elemwise_mul t u))
              args
        scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs,
               ("enable_float_output",) . showValue <$>
                 Anon.get #enable_float_output fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs,
               ("lhs_min",) . toRaw <$> Anon.get #lhs_min fullArgs,
               ("lhs_max",) . toRaw <$> Anon.get #lhs_max fullArgs,
               ("rhs_min",) . toRaw <$> Anon.get #rhs_min fullArgs,
               ("rhs_max",) . toRaw <$> Anon.get #rhs_max fullArgs]
      in
      applyRaw "_contrib_quantized_elemwise_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_embedding t u =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("weight", AttrOpt (t u)), '("min_weight", AttrOpt (t u)),
        '("max_weight", AttrOpt (t u))]

__contrib_quantized_embedding ::
                              forall t u r .
                                (Tensor t,
                                 FieldsAcc (ParameterList__contrib_quantized_embedding t u) r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__contrib_quantized_embedding args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_embedding t u))
              args
        scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$>
                 Just (Anon.get #input_dim fullArgs),
               ("output_dim",) . showValue <$>
                 Just (Anon.get #output_dim fullArgs),
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("sparse_grad",) . showValue <$> Anon.get #sparse_grad fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("min_weight",) . toRaw <$> Anon.get #min_weight fullArgs,
               ("max_weight",) . toRaw <$> Anon.get #max_weight fullArgs]
      in
      applyRaw "_contrib_quantized_embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_flatten t u =
     '[ '("_data", AttrReq (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_flatten ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_quantized_flatten t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_quantized_flatten args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_flatten t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs]
      in
      applyRaw "_contrib_quantized_flatten" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_fully_connected t u =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u)),
        '("min_data", AttrOpt (t u)), '("max_data", AttrOpt (t u)),
        '("min_weight", AttrOpt (t u)), '("max_weight", AttrOpt (t u)),
        '("min_bias", AttrOpt (t u)), '("max_bias", AttrOpt (t u))]

__contrib_quantized_fully_connected ::
                                    forall t u r .
                                      (Tensor t,
                                       FieldsAcc
                                         (ParameterList__contrib_quantized_fully_connected t u)
                                         r,
                                       HasCallStack, DType u) =>
                                      Record r -> TensorApply (t u)
__contrib_quantized_fully_connected args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_fully_connected t u))
              args
        scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 Just (Anon.get #num_hidden fullArgs),
               ("no_bias",) . showValue <$> Anon.get #no_bias fullArgs,
               ("flatten",) . showValue <$> Anon.get #flatten fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("bias",) . toRaw <$> Anon.get #bias fullArgs,
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs,
               ("min_weight",) . toRaw <$> Anon.get #min_weight fullArgs,
               ("max_weight",) . toRaw <$> Anon.get #max_weight fullArgs,
               ("min_bias",) . toRaw <$> Anon.get #min_bias fullArgs,
               ("max_bias",) . toRaw <$> Anon.get #max_bias fullArgs]
      in
      applyRaw "_contrib_quantized_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_pooling t u =
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
        '("_data", AttrReq (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_pooling ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__contrib_quantized_pooling t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__contrib_quantized_pooling args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_pooling t u))
              args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Anon.get #kernel fullArgs,
               ("pool_type",) . showValue <$> Anon.get #pool_type fullArgs,
               ("global_pool",) . showValue <$> Anon.get #global_pool fullArgs,
               ("cudnn_off",) . showValue <$> Anon.get #cudnn_off fullArgs,
               ("pooling_convention",) . showValue <$>
                 Anon.get #pooling_convention fullArgs,
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("p_value",) . showValue <$> Anon.get #p_value fullArgs,
               ("count_include_pad",) . showValue <$>
                 Anon.get #count_include_pad fullArgs,
               ("layout",) . showValue <$> Anon.get #layout fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_data",) . toRaw <$> Anon.get #min_data fullArgs,
               ("max_data",) . toRaw <$> Anon.get #max_data fullArgs]
      in
      applyRaw "_contrib_quantized_pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_quantized_rnn t u =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("projection_size", AttrOpt (Maybe Int)),
        '("lstm_state_clip_min", AttrOpt (Maybe Double)),
        '("lstm_state_clip_max", AttrOpt (Maybe Double)),
        '("lstm_state_clip_nan", AttrOpt Bool),
        '("use_sequence_length", AttrOpt Bool), '("_data", AttrReq (t u)),
        '("parameters", AttrOpt (t u)), '("state", AttrOpt (t u)),
        '("state_cell", AttrOpt (t u)), '("data_scale", AttrOpt (t u)),
        '("data_shift", AttrOpt (t u))]

__contrib_quantized_rnn ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__contrib_quantized_rnn t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__contrib_quantized_rnn args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_quantized_rnn t u))
              args
        scalarArgs
          = catMaybes
              [("state_size",) . showValue <$>
                 Just (Anon.get #state_size fullArgs),
               ("num_layers",) . showValue <$>
                 Just (Anon.get #num_layers fullArgs),
               ("bidirectional",) . showValue <$>
                 Anon.get #bidirectional fullArgs,
               ("mode",) . showValue <$> Just (Anon.get #mode fullArgs),
               ("p",) . showValue <$> Anon.get #p fullArgs,
               ("state_outputs",) . showValue <$>
                 Anon.get #state_outputs fullArgs,
               ("projection_size",) . showValue <$>
                 Anon.get #projection_size fullArgs,
               ("lstm_state_clip_min",) . showValue <$>
                 Anon.get #lstm_state_clip_min fullArgs,
               ("lstm_state_clip_max",) . showValue <$>
                 Anon.get #lstm_state_clip_max fullArgs,
               ("lstm_state_clip_nan",) . showValue <$>
                 Anon.get #lstm_state_clip_nan fullArgs,
               ("use_sequence_length",) . showValue <$>
                 Anon.get #use_sequence_length fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("parameters",) . toRaw <$> Anon.get #parameters fullArgs,
               ("state",) . toRaw <$> Anon.get #state fullArgs,
               ("state_cell",) . toRaw <$> Anon.get #state_cell fullArgs,
               ("data_scale",) . toRaw <$> Anon.get #data_scale fullArgs,
               ("data_shift",) . toRaw <$> Anon.get #data_shift fullArgs]
      in
      applyRaw "_contrib_quantized_rnn" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_requantize t u =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("_data", AttrReq (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_requantize ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__contrib_requantize t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__contrib_requantize args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_requantize t u))
              args
        scalarArgs
          = catMaybes
              [("out_type",) . showValue <$> Anon.get #out_type fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("min_range",) . toRaw <$> Anon.get #min_range fullArgs,
               ("max_range",) . toRaw <$> Anon.get #max_range fullArgs]
      in
      applyRaw "_contrib_requantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_round_ste t u =
     '[ '("_data", AttrReq (t u))]

__contrib_round_ste ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__contrib_round_ste t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__contrib_round_ste args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_round_ste t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_round_ste" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__contrib_sign_ste t u =
     '[ '("_data", AttrReq (t u))]

__contrib_sign_ste ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__contrib_sign_ste t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__contrib_sign_ste args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__contrib_sign_ste t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_contrib_sign_ste" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__copy t u = '[ '("_data", AttrReq (t u))]

__copy ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList__copy t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
__copy args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__copy t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__copyto t u = '[ '("_data", AttrReq (t u))]

__copyto ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__copyto t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__copyto args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__copyto t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_copyto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__cvcopyMakeBorder t u =
     '[ '("top", AttrReq Int), '("bot", AttrReq Int),
        '("left", AttrReq Int), '("right", AttrReq Int),
        '("_type", AttrOpt Int), '("value", AttrOpt Double),
        '("values", AttrOpt [Double]), '("src", AttrOpt (t u))]

__cvcopyMakeBorder ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__cvcopyMakeBorder t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__cvcopyMakeBorder args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__cvcopyMakeBorder t u))
              args
        scalarArgs
          = catMaybes
              [("top",) . showValue <$> Just (Anon.get #top fullArgs),
               ("bot",) . showValue <$> Just (Anon.get #bot fullArgs),
               ("left",) . showValue <$> Just (Anon.get #left fullArgs),
               ("right",) . showValue <$> Just (Anon.get #right fullArgs),
               ("type",) . showValue <$> Anon.get #_type fullArgs,
               ("value",) . showValue <$> Anon.get #value fullArgs,
               ("values",) . showValue <$> Anon.get #values fullArgs]
        tensorKeyArgs
          = catMaybes [("src",) . toRaw <$> Anon.get #src fullArgs]
      in
      applyRaw "_cvcopyMakeBorder" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__cvimdecode t u =
     '[ '("flag", AttrOpt Int), '("to_rgb", AttrOpt Bool),
        '("buf", AttrOpt (t u))]

__cvimdecode ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__cvimdecode t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__cvimdecode args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__cvimdecode t u))
              args
        scalarArgs
          = catMaybes
              [("flag",) . showValue <$> Anon.get #flag fullArgs,
               ("to_rgb",) . showValue <$> Anon.get #to_rgb fullArgs]
        tensorKeyArgs
          = catMaybes [("buf",) . toRaw <$> Anon.get #buf fullArgs]
      in
      applyRaw "_cvimdecode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__cvimread =
     '[ '("filename", AttrReq Text), '("flag", AttrOpt Int),
        '("to_rgb", AttrOpt Bool)]

__cvimread ::
           forall t u r .
             (Tensor t, FieldsAcc ParameterList__cvimread r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__cvimread args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__cvimread)) args
        scalarArgs
          = catMaybes
              [("filename",) . showValue <$> Just (Anon.get #filename fullArgs),
               ("flag",) . showValue <$> Anon.get #flag fullArgs,
               ("to_rgb",) . showValue <$> Anon.get #to_rgb fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_cvimread" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__cvimresize t u =
     '[ '("w", AttrReq Int), '("h", AttrReq Int),
        '("interp", AttrOpt Int), '("src", AttrOpt (t u))]

__cvimresize ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__cvimresize t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__cvimresize args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__cvimresize t u))
              args
        scalarArgs
          = catMaybes
              [("w",) . showValue <$> Just (Anon.get #w fullArgs),
               ("h",) . showValue <$> Just (Anon.get #h fullArgs),
               ("interp",) . showValue <$> Anon.get #interp fullArgs]
        tensorKeyArgs
          = catMaybes [("src",) . toRaw <$> Anon.get #src fullArgs]
      in
      applyRaw "_cvimresize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__div_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__div_scalar ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__div_scalar t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__div_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__div_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_div_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__equal ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList__equal t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
__equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__equal t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__equal_scalar ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__equal_scalar t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__equal_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__full =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"])),
        '("value", AttrReq Double)]

__full ::
       forall t u r .
         (Tensor t, FieldsAcc ParameterList__full r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
__full args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__full)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("value",) . showValue <$> Just (Anon.get #value fullArgs)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_full" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__grad_add t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__grad_add ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__grad_add t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__grad_add args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__grad_add t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_grad_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__greater t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__greater ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__greater t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__greater args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__greater t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__greater_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__greater_equal ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__greater_equal t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__greater_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__greater_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__greater_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__greater_equal_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__greater_equal_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__greater_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__greater_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_greater_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__greater_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__greater_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__greater_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__greater_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__greater_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_greater_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__histogram t u =
     '[ '("bin_cnt", AttrOpt (Maybe Int)), '("range", AttrOpt Int),
        '("_data", AttrReq (t u)), '("bins", AttrOpt (t u))]

__histogram ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__histogram t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__histogram args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__histogram t u)) args
        scalarArgs
          = catMaybes
              [("bin_cnt",) . showValue <$> Anon.get #bin_cnt fullArgs,
               ("range",) . showValue <$> Anon.get #range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("bins",) . toRaw <$> Anon.get #bins fullArgs]
      in
      applyRaw "_histogram" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__hypot t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__hypot ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList__hypot t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
__hypot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__hypot t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__hypot_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__hypot_scalar ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__hypot_scalar t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__hypot_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__hypot_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_hypot_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__identity_with_attr_like_rhs t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__identity_with_attr_like_rhs ::
                              forall t u r .
                                (Tensor t,
                                 FieldsAcc (ParameterList__identity_with_attr_like_rhs t u) r,
                                 HasCallStack, DType u) =>
                                Record r -> TensorApply (t u)
__identity_with_attr_like_rhs args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__identity_with_attr_like_rhs t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_identity_with_attr_like_rhs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_adjust_lighting t u =
     '[ '("alpha", AttrReq [Float]), '("_data", AttrReq (t u))]

__image_adjust_lighting ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__image_adjust_lighting t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__image_adjust_lighting args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_adjust_lighting t u))
              args
        scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> Just (Anon.get #alpha fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_adjust_lighting" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_crop t u =
     '[ '("x", AttrReq Int), '("y", AttrReq Int),
        '("width", AttrReq Int), '("height", AttrReq Int),
        '("_data", AttrReq (t u))]

__image_crop ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__image_crop t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__image_crop args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__image_crop t u))
              args
        scalarArgs
          = catMaybes
              [("x",) . showValue <$> Just (Anon.get #x fullArgs),
               ("y",) . showValue <$> Just (Anon.get #y fullArgs),
               ("width",) . showValue <$> Just (Anon.get #width fullArgs),
               ("height",) . showValue <$> Just (Anon.get #height fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_flip_left_right t u =
     '[ '("_data", AttrReq (t u))]

__image_flip_left_right ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__image_flip_left_right t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__image_flip_left_right args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_flip_left_right t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_flip_left_right" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_flip_top_bottom t u =
     '[ '("_data", AttrReq (t u))]

__image_flip_top_bottom ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__image_flip_top_bottom t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__image_flip_top_bottom args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_flip_top_bottom t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_flip_top_bottom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_normalize t u =
     '[ '("mean", AttrOpt [Float]), '("std", AttrOpt [Float]),
        '("_data", AttrReq (t u))]

__image_normalize ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__image_normalize t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__image_normalize args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_normalize t u))
              args
        scalarArgs
          = catMaybes
              [("mean",) . showValue <$> Anon.get #mean fullArgs,
               ("std",) . showValue <$> Anon.get #std fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_normalize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_brightness t u =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("_data", AttrReq (t u))]

__image_random_brightness ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__image_random_brightness t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__image_random_brightness args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_brightness t u))
              args
        scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 Just (Anon.get #min_factor fullArgs),
               ("max_factor",) . showValue <$>
                 Just (Anon.get #max_factor fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_brightness" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_color_jitter t u =
     '[ '("brightness", AttrReq Float), '("contrast", AttrReq Float),
        '("saturation", AttrReq Float), '("hue", AttrReq Float),
        '("_data", AttrReq (t u))]

__image_random_color_jitter ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList__image_random_color_jitter t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__image_random_color_jitter args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_color_jitter t u))
              args
        scalarArgs
          = catMaybes
              [("brightness",) . showValue <$>
                 Just (Anon.get #brightness fullArgs),
               ("contrast",) . showValue <$> Just (Anon.get #contrast fullArgs),
               ("saturation",) . showValue <$>
                 Just (Anon.get #saturation fullArgs),
               ("hue",) . showValue <$> Just (Anon.get #hue fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_color_jitter" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_contrast t u =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("_data", AttrReq (t u))]

__image_random_contrast ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__image_random_contrast t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__image_random_contrast args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_contrast t u))
              args
        scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 Just (Anon.get #min_factor fullArgs),
               ("max_factor",) . showValue <$>
                 Just (Anon.get #max_factor fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_contrast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_flip_left_right t u =
     '[ '("_data", AttrReq (t u))]

__image_random_flip_left_right ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__image_random_flip_left_right t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__image_random_flip_left_right args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_flip_left_right t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_flip_left_right" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_flip_top_bottom t u =
     '[ '("_data", AttrReq (t u))]

__image_random_flip_top_bottom ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__image_random_flip_top_bottom t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__image_random_flip_top_bottom args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_flip_top_bottom t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_flip_top_bottom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_hue t u =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("_data", AttrReq (t u))]

__image_random_hue ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__image_random_hue t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__image_random_hue args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_hue t u))
              args
        scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 Just (Anon.get #min_factor fullArgs),
               ("max_factor",) . showValue <$>
                 Just (Anon.get #max_factor fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_hue" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_lighting t u =
     '[ '("alpha_std", AttrOpt Float), '("_data", AttrReq (t u))]

__image_random_lighting ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__image_random_lighting t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__image_random_lighting args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_lighting t u))
              args
        scalarArgs
          = catMaybes
              [("alpha_std",) . showValue <$> Anon.get #alpha_std fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_lighting" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_random_saturation t u =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("_data", AttrReq (t u))]

__image_random_saturation ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__image_random_saturation t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__image_random_saturation args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_random_saturation t u))
              args
        scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 Just (Anon.get #min_factor fullArgs),
               ("max_factor",) . showValue <$>
                 Just (Anon.get #max_factor fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_random_saturation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_resize t u =
     '[ '("size", AttrOpt [Int]), '("keep_ratio", AttrOpt Bool),
        '("interp", AttrOpt Int), '("_data", AttrReq (t u))]

__image_resize ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__image_resize t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__image_resize args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__image_resize t u))
              args
        scalarArgs
          = catMaybes
              [("size",) . showValue <$> Anon.get #size fullArgs,
               ("keep_ratio",) . showValue <$> Anon.get #keep_ratio fullArgs,
               ("interp",) . showValue <$> Anon.get #interp fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_resize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__image_to_tensor t u =
     '[ '("_data", AttrReq (t u))]

__image_to_tensor ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__image_to_tensor t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__image_to_tensor args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__image_to_tensor t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_image_to_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__imdecode t u =
     '[ '("index", AttrOpt Int), '("x0", AttrOpt Int),
        '("y0", AttrOpt Int), '("x1", AttrOpt Int), '("y1", AttrOpt Int),
        '("c", AttrOpt Int), '("size", AttrOpt Int),
        '("mean", AttrOpt (t u))]

__imdecode ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__imdecode t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__imdecode args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__imdecode t u)) args
        scalarArgs
          = catMaybes
              [("index",) . showValue <$> Anon.get #index fullArgs,
               ("x0",) . showValue <$> Anon.get #x0 fullArgs,
               ("y0",) . showValue <$> Anon.get #y0 fullArgs,
               ("x1",) . showValue <$> Anon.get #x1 fullArgs,
               ("y1",) . showValue <$> Anon.get #y1 fullArgs,
               ("c",) . showValue <$> Anon.get #c fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs]
        tensorKeyArgs
          = catMaybes [("mean",) . toRaw <$> Anon.get #mean fullArgs]
      in
      applyRaw "_imdecode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__lesser t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__lesser ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__lesser t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__lesser args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__lesser t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_lesser" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__lesser_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__lesser_equal ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__lesser_equal t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__lesser_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__lesser_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_lesser_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__lesser_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__lesser_equal_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__lesser_equal_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__lesser_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__lesser_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_lesser_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__lesser_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__lesser_scalar ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__lesser_scalar t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__lesser_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__lesser_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_lesser_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_det t u = '[ '("a", AttrOpt (t u))]

__linalg_det ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__linalg_det t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__linalg_det args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_det t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_det" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_extractdiag t u =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt (t u))]

__linalg_extractdiag ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__linalg_extractdiag t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__linalg_extractdiag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__linalg_extractdiag t u))
              args
        scalarArgs
          = catMaybes [("offset",) . showValue <$> Anon.get #offset fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_extractdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_extracttrian t u =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__linalg_extracttrian ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__linalg_extracttrian t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__linalg_extracttrian args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__linalg_extracttrian t u))
              args
        scalarArgs
          = catMaybes
              [("offset",) . showValue <$> Anon.get #offset fullArgs,
               ("lower",) . showValue <$> Anon.get #lower fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_extracttrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_gelqf t u = '[ '("a", AttrOpt (t u))]

__linalg_gelqf ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__linalg_gelqf t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__linalg_gelqf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_gelqf t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_gelqf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_gemm t u =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("beta", AttrOpt Double),
        '("axis", AttrOpt Int), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u)), '("c", AttrOpt (t u))]

__linalg_gemm ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__linalg_gemm t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__linalg_gemm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_gemm t u))
              args
        scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$> Anon.get #transpose_a fullArgs,
               ("transpose_b",) . showValue <$> Anon.get #transpose_b fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs,
               ("c",) . toRaw <$> Anon.get #c fullArgs]
      in
      applyRaw "_linalg_gemm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_gemm2 t u =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("axis", AttrOpt Int),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_gemm2 ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__linalg_gemm2 t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__linalg_gemm2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_gemm2 t u))
              args
        scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$> Anon.get #transpose_a fullArgs,
               ("transpose_b",) . showValue <$> Anon.get #transpose_b fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_linalg_gemm2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_inverse t u = '[ '("a", AttrOpt (t u))]

__linalg_inverse ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__linalg_inverse t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__linalg_inverse args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_inverse t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_inverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_makediag t u =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt (t u))]

__linalg_makediag ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__linalg_makediag t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__linalg_makediag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__linalg_makediag t u))
              args
        scalarArgs
          = catMaybes [("offset",) . showValue <$> Anon.get #offset fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_makediag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_maketrian t u =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__linalg_maketrian ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__linalg_maketrian t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__linalg_maketrian args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__linalg_maketrian t u))
              args
        scalarArgs
          = catMaybes
              [("offset",) . showValue <$> Anon.get #offset fullArgs,
               ("lower",) . showValue <$> Anon.get #lower fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_maketrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_potrf t u = '[ '("a", AttrOpt (t u))]

__linalg_potrf ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__linalg_potrf t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__linalg_potrf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_potrf t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_potrf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_potri t u = '[ '("a", AttrOpt (t u))]

__linalg_potri ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__linalg_potri t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__linalg_potri args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_potri t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_potri" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_slogdet t u = '[ '("a", AttrOpt (t u))]

__linalg_slogdet ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__linalg_slogdet t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__linalg_slogdet args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_slogdet t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_slogdet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_sumlogdiag t u =
     '[ '("a", AttrOpt (t u))]

__linalg_sumlogdiag ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__linalg_sumlogdiag t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__linalg_sumlogdiag args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__linalg_sumlogdiag t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_sumlogdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_syevd t u = '[ '("a", AttrOpt (t u))]

__linalg_syevd ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__linalg_syevd t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__linalg_syevd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_syevd t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_syevd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_syrk t u =
     '[ '("transpose", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u))]

__linalg_syrk ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__linalg_syrk t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__linalg_syrk args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_syrk t u))
              args
        scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> Anon.get #transpose fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_linalg_syrk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_trmm t u =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_trmm ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__linalg_trmm t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__linalg_trmm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_trmm t u))
              args
        scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> Anon.get #transpose fullArgs,
               ("rightside",) . showValue <$> Anon.get #rightside fullArgs,
               ("lower",) . showValue <$> Anon.get #lower fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_linalg_trmm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linalg_trsm t u =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_trsm ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__linalg_trsm t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__linalg_trsm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linalg_trsm t u))
              args
        scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> Anon.get #transpose fullArgs,
               ("rightside",) . showValue <$> Anon.get #rightside fullArgs,
               ("lower",) . showValue <$> Anon.get #lower fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_linalg_trsm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__linspace =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__linspace ::
           forall t u r .
             (Tensor t, FieldsAcc ParameterList__linspace r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__linspace args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__linspace)) args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Just (Anon.get #start fullArgs),
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("repeat",) . showValue <$> Anon.get #repeat fullArgs,
               ("infer_range",) . showValue <$> Anon.get #infer_range fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_linspace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_and t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_and ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__logical_and t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__logical_and args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__logical_and t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_logical_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_and_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__logical_and_scalar ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__logical_and_scalar t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__logical_and_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__logical_and_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_logical_and_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_or t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_or ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__logical_or t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__logical_or args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__logical_or t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_logical_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_or_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__logical_or_scalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__logical_or_scalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__logical_or_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__logical_or_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_logical_or_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_xor t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_xor ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__logical_xor t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__logical_xor args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__logical_xor t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_logical_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__logical_xor_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__logical_xor_scalar ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__logical_xor_scalar t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__logical_xor_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__logical_xor_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_logical_xor_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__maximum t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__maximum ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__maximum t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__maximum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__maximum t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__maximum_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__maximum_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__maximum_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__maximum_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__maximum_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_maximum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__minimum t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__minimum ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__minimum t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__minimum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__minimum t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__minimum_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__minimum_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__minimum_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__minimum_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__minimum_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_minimum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__minus_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__minus_scalar ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__minus_scalar t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__minus_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__minus_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_minus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__mod t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__mod ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList__mod t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
__mod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__mod t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__mod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__mod_scalar ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__mod_scalar t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__mod_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__mod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__mp_adamw_update t u =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("weight32", AttrOpt (t u)),
        '("rescale_grad", AttrOpt (t u))]

__mp_adamw_update ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__mp_adamw_update t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__mp_adamw_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__mp_adamw_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("eta",) . showValue <$> Just (Anon.get #eta fullArgs),
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mean",) . toRaw <$> Anon.get #mean fullArgs,
               ("var",) . toRaw <$> Anon.get #var fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs,
               ("rescale_grad",) . toRaw <$> Anon.get #rescale_grad fullArgs]
      in
      applyRaw "_mp_adamw_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__mul_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__mul_scalar ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__mul_scalar t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__mul_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__mul_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_mul_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__multi_adamw_update t u =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

__multi_adamw_update ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__multi_adamw_update t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__multi_adamw_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__multi_adamw_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("etas",) . showValue <$> Just (Anon.get #etas fullArgs),
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_multi_adamw_update" scalarArgs (Right tensorVarArgs)

type ParameterList__multi_lamb_update t u =
     '[ '("learning_rates", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("rescale_grad", AttrOpt Float),
        '("lower_bound", AttrOpt Float), '("upper_bound", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("bias_correction", AttrOpt Bool), '("step_count", AttrReq [Int]),
        '("num_tensors", AttrOpt Int), '("_data", AttrReq [t u])]

__multi_lamb_update ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__multi_lamb_update t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__multi_lamb_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__multi_lamb_update t u))
              args
        scalarArgs
          = catMaybes
              [("learning_rates",) . showValue <$>
                 Just (Anon.get #learning_rates fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("lower_bound",) . showValue <$> Anon.get #lower_bound fullArgs,
               ("upper_bound",) . showValue <$> Anon.get #upper_bound fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("bias_correction",) . showValue <$>
                 Anon.get #bias_correction fullArgs,
               ("step_count",) . showValue <$>
                 Just (Anon.get #step_count fullArgs),
               ("num_tensors",) . showValue <$> Anon.get #num_tensors fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_multi_lamb_update" scalarArgs (Right tensorVarArgs)

type ParameterList__multi_mp_adamw_update t u =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

__multi_mp_adamw_update ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__multi_mp_adamw_update t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__multi_mp_adamw_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__multi_mp_adamw_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("etas",) . showValue <$> Just (Anon.get #etas fullArgs),
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "_multi_mp_adamw_update" scalarArgs (Right tensorVarArgs)

type ParameterList__multi_mp_lamb_update t u =
     '[ '("learning_rates", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("rescale_grad", AttrOpt Float),
        '("lower_bound", AttrOpt Float), '("upper_bound", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("bias_correction", AttrOpt Bool), '("step_count", AttrReq [Int]),
        '("num_tensors", AttrOpt Int), '("_data", AttrReq [t u])]

__multi_mp_lamb_update ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__multi_mp_lamb_update t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__multi_mp_lamb_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__multi_mp_lamb_update t u))
              args
        scalarArgs
          = catMaybes
              [("learning_rates",) . showValue <$>
                 Just (Anon.get #learning_rates fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("lower_bound",) . showValue <$> Anon.get #lower_bound fullArgs,
               ("upper_bound",) . showValue <$> Anon.get #upper_bound fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("bias_correction",) . showValue <$>
                 Anon.get #bias_correction fullArgs,
               ("step_count",) . showValue <$>
                 Just (Anon.get #step_count fullArgs),
               ("num_tensors",) . showValue <$> Anon.get #num_tensors fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "_multi_mp_lamb_update" scalarArgs (Right tensorVarArgs)

type ParameterList__not_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__not_equal ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__not_equal t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__not_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__not_equal t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__not_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__not_equal_scalar ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__not_equal_scalar t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__not_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__not_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_not_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_all t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__np_all ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_all t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_all args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_all t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_all" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_any t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__np_any ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_any t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_any args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_any t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_any" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_atleast_1d t u =
     '[ '("num_args", AttrReq Int), '("arys", AttrReq [t u])]

__np_atleast_1d ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__np_atleast_1d t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__np_atleast_1d args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_atleast_1d t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #arys fullArgs :: [RawTensor t]
      in applyRaw "_np_atleast_1d" scalarArgs (Right tensorVarArgs)

type ParameterList__np_atleast_2d t u =
     '[ '("num_args", AttrReq Int), '("arys", AttrReq [t u])]

__np_atleast_2d ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__np_atleast_2d t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__np_atleast_2d args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_atleast_2d t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #arys fullArgs :: [RawTensor t]
      in applyRaw "_np_atleast_2d" scalarArgs (Right tensorVarArgs)

type ParameterList__np_atleast_3d t u =
     '[ '("num_args", AttrReq Int), '("arys", AttrReq [t u])]

__np_atleast_3d ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__np_atleast_3d t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__np_atleast_3d args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_atleast_3d t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #arys fullArgs :: [RawTensor t]
      in applyRaw "_np_atleast_3d" scalarArgs (Right tensorVarArgs)

type ParameterList__np_copy t u = '[ '("a", AttrOpt (t u))]

__np_copy ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__np_copy t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__np_copy args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_copy t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_cumsum t u =
     '[ '("axis", AttrOpt (Maybe Int)),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("a", AttrOpt (t u))]

__np_cumsum ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__np_cumsum t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__np_cumsum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_cumsum t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_cumsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_diag t u =
     '[ '("k", AttrOpt Int), '("_data", AttrReq (t u))]

__np_diag ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__np_diag t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__np_diag args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_diag t u)) args
        scalarArgs
          = catMaybes [("k",) . showValue <$> Anon.get #k fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_diagflat t u =
     '[ '("k", AttrOpt Int), '("_data", AttrReq (t u))]

__np_diagflat ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__np_diagflat t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__np_diagflat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_diagflat t u))
              args
        scalarArgs
          = catMaybes [("k",) . showValue <$> Anon.get #k fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_diagflat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_diagonal t u =
     '[ '("offset", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("_data", AttrReq (t u))]

__np_diagonal ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__np_diagonal t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__np_diagonal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_diagonal t u))
              args
        scalarArgs
          = catMaybes
              [("offset",) . showValue <$> Anon.get #offset fullArgs,
               ("axis1",) . showValue <$> Anon.get #axis1 fullArgs,
               ("axis2",) . showValue <$> Anon.get #axis2 fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_diagonal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_dot t u =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__np_dot ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_dot t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_dot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_dot t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_np_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_max t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt (t u))]

__np_max ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_max t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_max args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_max t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("initial",) . showValue <$> Anon.get #initial fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_min t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt (t u))]

__np_min ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_min t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_min args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_min t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("initial",) . showValue <$> Anon.get #initial fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_moveaxis t u =
     '[ '("source", AttrReq [Int]), '("destination", AttrReq [Int]),
        '("a", AttrOpt (t u))]

__np_moveaxis ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__np_moveaxis t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__np_moveaxis args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_moveaxis t u))
              args
        scalarArgs
          = catMaybes
              [("source",) . showValue <$> Just (Anon.get #source fullArgs),
               ("destination",) . showValue <$>
                 Just (Anon.get #destination fullArgs)]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_moveaxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_prod t u =
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
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__np_prod t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__np_prod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_prod t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("initial",) . showValue <$> Anon.get #initial fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_reshape t u =
     '[ '("newshape", AttrReq [Int]), '("order", AttrOpt Text),
        '("a", AttrOpt (t u))]

__np_reshape ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__np_reshape t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__np_reshape args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_reshape t u))
              args
        scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> Just (Anon.get #newshape fullArgs),
               ("order",) . showValue <$> Anon.get #order fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_roll t u =
     '[ '("shift", AttrOpt (Maybe [Int])),
        '("axis", AttrOpt (Maybe [Int])), '("_data", AttrReq (t u))]

__np_roll ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__np_roll t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__np_roll args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_roll t u)) args
        scalarArgs
          = catMaybes
              [("shift",) . showValue <$> Anon.get #shift fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_roll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_squeeze t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("a", AttrOpt (t u))]

__np_squeeze ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__np_squeeze t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__np_squeeze args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_squeeze t u))
              args
        scalarArgs
          = catMaybes [("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_sum t u =
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
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList__np_sum t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
__np_sum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_sum t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("initial",) . showValue <$> Anon.get #initial fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_trace t u =
     '[ '("offset", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("_data", AttrReq (t u))]

__np_trace ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__np_trace t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__np_trace args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_trace t u)) args
        scalarArgs
          = catMaybes
              [("offset",) . showValue <$> Anon.get #offset fullArgs,
               ("axis1",) . showValue <$> Anon.get #axis1 fullArgs,
               ("axis2",) . showValue <$> Anon.get #axis2 fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_np_trace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__np_transpose t u =
     '[ '("axes", AttrOpt [Int]), '("a", AttrOpt (t u))]

__np_transpose ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__np_transpose t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__np_transpose args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__np_transpose t u))
              args
        scalarArgs
          = catMaybes [("axes",) . showValue <$> Anon.get #axes fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_np_transpose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_absolute t u = '[ '("x", AttrOpt (t u))]

__npi_absolute ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_absolute t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_absolute args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_absolute t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_absolute" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_add t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_add ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_add t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_add args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_add t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_add_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_add_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_add_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_add_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_add_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_add_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arange =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_arange ::
             forall t u r .
               (Tensor t, FieldsAcc ParameterList__npi_arange r, HasCallStack,
                DType u) =>
               Record r -> TensorApply (t u)
__npi_arange args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arange)) args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Just (Anon.get #start fullArgs),
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("repeat",) . showValue <$> Anon.get #repeat fullArgs,
               ("infer_range",) . showValue <$> Anon.get #infer_range fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_arange" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arccos t u = '[ '("x", AttrOpt (t u))]

__npi_arccos ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_arccos t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_arccos args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arccos t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arccosh t u = '[ '("x", AttrOpt (t u))]

__npi_arccosh ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_arccosh t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_arccosh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arccosh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arcsin t u = '[ '("x", AttrOpt (t u))]

__npi_arcsin ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_arcsin t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_arcsin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arcsin t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arcsinh t u = '[ '("x", AttrOpt (t u))]

__npi_arcsinh ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_arcsinh t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_arcsinh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arcsinh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arctan t u = '[ '("x", AttrOpt (t u))]

__npi_arctan ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_arctan t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_arctan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arctan t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arctan2 t u =
     '[ '("x1", AttrOpt (t u)), '("x2", AttrOpt (t u))]

__npi_arctan2 ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_arctan2 t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_arctan2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arctan2 t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) . toRaw <$> Anon.get #x1 fullArgs,
               ("x2",) . toRaw <$> Anon.get #x2 fullArgs]
      in
      applyRaw "_npi_arctan2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arctan2_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_arctan2_scalar ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__npi_arctan2_scalar t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__npi_arctan2_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_arctan2_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_arctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_arctanh t u = '[ '("x", AttrOpt (t u))]

__npi_arctanh ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_arctanh t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_arctanh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_arctanh t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_argmax t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_argmax ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_argmax t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t Int64)
__npi_argmax args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_argmax t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_argmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_argmin t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_argmin ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_argmin t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t Int64)
__npi_argmin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_argmin t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_argmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_around t u =
     '[ '("decimals", AttrOpt Int), '("x", AttrOpt (t u))]

__npi_around ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_around t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_around args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_around t u))
              args
        scalarArgs
          = catMaybes
              [("decimals",) . showValue <$> Anon.get #decimals fullArgs]
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_around" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_average t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("returned", AttrOpt Bool),
        '("weighted", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("weights", AttrOpt (t u))]

__npi_average ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_average t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_average args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_average t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("returned",) . showValue <$> Anon.get #returned fullArgs,
               ("weighted",) . showValue <$> Anon.get #weighted fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("weights",) . toRaw <$> Anon.get #weights fullArgs]
      in
      applyRaw "_npi_average" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_backward_ediff1d = '[]

__npi_backward_ediff1d ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__npi_backward_ediff1d r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__npi_backward_ediff1d args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_backward_ediff1d))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_ediff1d" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_backward_nan_to_num = '[]

__npi_backward_nan_to_num ::
                          forall t u r .
                            (Tensor t, FieldsAcc ParameterList__npi_backward_nan_to_num r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__npi_backward_nan_to_num args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_backward_nan_to_num))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_nan_to_num" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_backward_polyval = '[]

__npi_backward_polyval ::
                       forall t u r .
                         (Tensor t, FieldsAcc ParameterList__npi_backward_polyval r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__npi_backward_polyval args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_backward_polyval))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_polyval" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bernoulli t u =
     '[ '("prob", AttrReq (Maybe Float)),
        '("logit", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "uint8"])),
        '("is_logit", AttrReq Bool), '("input1", AttrOpt (t u))]

__npi_bernoulli ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_bernoulli t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__npi_bernoulli args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_bernoulli t u))
              args
        scalarArgs
          = catMaybes
              [("prob",) . showValue <$> Just (Anon.get #prob fullArgs),
               ("logit",) . showValue <$> Just (Anon.get #logit fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("is_logit",) . showValue <$> Just (Anon.get #is_logit fullArgs)]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_bernoulli" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bincount t u =
     '[ '("minlength", AttrOpt Int), '("has_weights", AttrOpt Bool),
        '("_data", AttrReq (t u)), '("weights", AttrOpt (t u))]

__npi_bincount ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_bincount t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_bincount args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_bincount t u))
              args
        scalarArgs
          = catMaybes
              [("minlength",) . showValue <$> Anon.get #minlength fullArgs,
               ("has_weights",) . showValue <$> Anon.get #has_weights fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("weights",) . toRaw <$> Anon.get #weights fullArgs]
      in
      applyRaw "_npi_bincount" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_and t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_and ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_bitwise_and t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_bitwise_and args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_and t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_bitwise_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_and_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_bitwise_and_scalar ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__npi_bitwise_and_scalar t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__npi_bitwise_and_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_and_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_bitwise_and_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_not t u = '[ '("x", AttrOpt (t u))]

__npi_bitwise_not ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_bitwise_not t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_bitwise_not args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_not t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_bitwise_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_or t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_or ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_bitwise_or t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_bitwise_or args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_bitwise_or t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_bitwise_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_or_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_bitwise_or_scalar ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__npi_bitwise_or_scalar t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__npi_bitwise_or_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_or_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_bitwise_or_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_xor t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_xor ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_bitwise_xor t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_bitwise_xor args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_xor t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_bitwise_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_bitwise_xor_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_bitwise_xor_scalar ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__npi_bitwise_xor_scalar t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__npi_bitwise_xor_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_bitwise_xor_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_bitwise_xor_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_blackman =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_blackman ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__npi_blackman r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__npi_blackman args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_blackman)) args
        scalarArgs
          = catMaybes
              [("m",) . showValue <$> Anon.get #m fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_blackman" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_boolean_mask_assign_scalar t u =
     '[ '("value", AttrOpt Float), '("start_axis", AttrOpt Int),
        '("_data", AttrReq (t u)), '("mask", AttrOpt (t u))]

__npi_boolean_mask_assign_scalar ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__npi_boolean_mask_assign_scalar t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__npi_boolean_mask_assign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_boolean_mask_assign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("value",) . showValue <$> Anon.get #value fullArgs,
               ("start_axis",) . showValue <$> Anon.get #start_axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("mask",) . toRaw <$> Anon.get #mask fullArgs]
      in
      applyRaw "_npi_boolean_mask_assign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_boolean_mask_assign_tensor t u =
     '[ '("start_axis", AttrOpt Int), '("_data", AttrReq (t u)),
        '("mask", AttrOpt (t u)), '("value", AttrOpt (t u))]

__npi_boolean_mask_assign_tensor ::
                                 forall t u r .
                                   (Tensor t,
                                    FieldsAcc (ParameterList__npi_boolean_mask_assign_tensor t u) r,
                                    HasCallStack, DType u) =>
                                   Record r -> TensorApply (t u)
__npi_boolean_mask_assign_tensor args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_boolean_mask_assign_tensor t u))
              args
        scalarArgs
          = catMaybes
              [("start_axis",) . showValue <$> Anon.get #start_axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("mask",) . toRaw <$> Anon.get #mask fullArgs,
               ("value",) . toRaw <$> Anon.get #value fullArgs]
      in
      applyRaw "_npi_boolean_mask_assign_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_broadcast_to t u =
     '[ '("shape", AttrOpt [Int]), '("array", AttrOpt (t u))]

__npi_broadcast_to ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_broadcast_to t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_broadcast_to args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_broadcast_to t u))
              args
        scalarArgs
          = catMaybes [("shape",) . showValue <$> Anon.get #shape fullArgs]
        tensorKeyArgs
          = catMaybes [("array",) . toRaw <$> Anon.get #array fullArgs]
      in
      applyRaw "_npi_broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_cbrt t u = '[ '("x", AttrOpt (t u))]

__npi_cbrt ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_cbrt t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_cbrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_cbrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_ceil t u = '[ '("x", AttrOpt (t u))]

__npi_ceil ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_ceil t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_ceil args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_ceil t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_ceil" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_choice t u =
     '[ '("a", AttrReq Int), '("size", AttrReq Int),
        '("ctx", AttrOpt Text), '("replace", AttrOpt Bool),
        '("weighted", AttrOpt Bool), '("input1", AttrOpt (t u)),
        '("input2", AttrOpt (t u))]

__npi_choice ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_choice t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_choice args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_choice t u))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Just (Anon.get #a fullArgs),
               ("size",) . showValue <$> Just (Anon.get #size fullArgs),
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("replace",) . showValue <$> Anon.get #replace fullArgs,
               ("weighted",) . showValue <$> Anon.get #weighted fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_choice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_cholesky t u = '[ '("a", AttrOpt (t u))]

__npi_cholesky ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_cholesky t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_cholesky args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_cholesky t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_cholesky" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_column_stack t u =
     '[ '("num_args", AttrReq Int), '("_data", AttrReq [t u])]

__npi_column_stack ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_column_stack t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_column_stack args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_column_stack t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_column_stack" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_concatenate t u =
     '[ '("num_args", AttrReq Int), '("axis", AttrOpt Int),
        '("_data", AttrReq [t u])]

__npi_concatenate ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_concatenate t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_concatenate args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_concatenate t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_concatenate" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_copysign t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_copysign ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_copysign t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_copysign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_copysign t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_copysign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_copysign_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_copysign_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__npi_copysign_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__npi_copysign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_copysign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_copysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_cos t u = '[ '("x", AttrOpt (t u))]

__npi_cos ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_cos t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_cos args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_cos t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_cosh t u = '[ '("x", AttrOpt (t u))]

__npi_cosh ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_cosh t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_cosh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_cosh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_degrees t u = '[ '("x", AttrOpt (t u))]

__npi_degrees ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_degrees t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_degrees args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_degrees t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_delete t u =
     '[ '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("obj", AttrOpt (t u))]

__npi_delete ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_delete t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_delete args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_delete t u))
              args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Anon.get #start fullArgs,
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("int_ind",) . showValue <$> Anon.get #int_ind fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> Anon.get #arr fullArgs,
               ("obj",) . toRaw <$> Anon.get #obj fullArgs]
      in
      applyRaw "_npi_delete" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_diag_indices_from t u =
     '[ '("_data", AttrReq (t u))]

__npi_diag_indices_from ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__npi_diag_indices_from t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__npi_diag_indices_from args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_diag_indices_from t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_diag_indices_from" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_diff t u =
     '[ '("n", AttrOpt Int), '("axis", AttrOpt Int),
        '("a", AttrOpt (t u))]

__npi_diff ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_diff t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_diff args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_diff t u)) args
        scalarArgs
          = catMaybes
              [("n",) . showValue <$> Anon.get #n fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_diff" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_dsplit t u =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("_data", AttrReq (t u))]

__npi_dsplit ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_dsplit t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_dsplit args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_dsplit t u))
              args
        scalarArgs
          = catMaybes
              [("indices",) . showValue <$> Just (Anon.get #indices fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("squeeze_axis",) . showValue <$> Anon.get #squeeze_axis fullArgs,
               ("sections",) . showValue <$> Anon.get #sections fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_dsplit" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_dstack t u =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("_data", AttrReq [t u])]

__npi_dstack ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_dstack t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_dstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_dstack t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("dim",) . showValue <$> Anon.get #dim fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_dstack" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_ediff1d t u =
     '[ '("to_begin_arr_given", AttrOpt Bool),
        '("to_end_arr_given", AttrOpt Bool),
        '("to_begin_scalar", AttrOpt (Maybe Double)),
        '("to_end_scalar", AttrOpt (Maybe Double)),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u)),
        '("input3", AttrOpt (t u))]

__npi_ediff1d ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_ediff1d t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_ediff1d args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_ediff1d t u))
              args
        scalarArgs
          = catMaybes
              [("to_begin_arr_given",) . showValue <$>
                 Anon.get #to_begin_arr_given fullArgs,
               ("to_end_arr_given",) . showValue <$>
                 Anon.get #to_end_arr_given fullArgs,
               ("to_begin_scalar",) . showValue <$>
                 Anon.get #to_begin_scalar fullArgs,
               ("to_end_scalar",) . showValue <$>
                 Anon.get #to_end_scalar fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs,
               ("input3",) . toRaw <$> Anon.get #input3 fullArgs]
      in
      applyRaw "_npi_ediff1d" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_eig t u = '[ '("a", AttrOpt (t u))]

__npi_eig ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_eig t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_eig args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_eig t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_eig" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_eigh t u =
     '[ '("uPLO", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_eigh ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_eigh t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_eigh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_eigh t u)) args
        scalarArgs
          = catMaybes [("uPLO",) . showValue <$> Anon.get #uPLO fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_eigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_eigvals t u = '[ '("a", AttrOpt (t u))]

__npi_eigvals ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_eigvals t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_eigvals args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_eigvals t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_eigvals" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_eigvalsh t u =
     '[ '("uPLO", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_eigvalsh ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_eigvalsh t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_eigvalsh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_eigvalsh t u))
              args
        scalarArgs
          = catMaybes [("uPLO",) . showValue <$> Anon.get #uPLO fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_eigvalsh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_einsum t u =
     '[ '("num_args", AttrReq Int), '("subscripts", AttrOpt Text),
        '("optimize", AttrOpt Int), '("_data", AttrReq [t u])]

__npi_einsum ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_einsum t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_einsum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_einsum t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("subscripts",) . showValue <$> Anon.get #subscripts fullArgs,
               ("optimize",) . showValue <$> Anon.get #optimize fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_einsum" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_equal ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_equal t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t Bool)
__npi_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_equal t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_equal_scalar ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_equal_scalar t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t Bool)
__npi_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_exp t u = '[ '("x", AttrOpt (t u))]

__npi_exp ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_exp t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_exp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_exp t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_exp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_expm1 t u = '[ '("x", AttrOpt (t u))]

__npi_expm1 ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_expm1 t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_expm1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_expm1 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_exponential t u =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u))]

__npi_exponential ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_exponential t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_exponential args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_exponential t u))
              args
        scalarArgs
          = catMaybes
              [("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_fix t u = '[ '("x", AttrOpt (t u))]

__npi_fix ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_fix t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_fix args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_fix t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_fix" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_flip t u =
     '[ '("axis", AttrReq [Int]), '("_data", AttrReq (t u))]

__npi_flip ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_flip t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_flip args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_flip t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Just (Anon.get #axis fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_flip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_floor t u = '[ '("x", AttrOpt (t u))]

__npi_floor ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_floor t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_floor args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_floor t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_floor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_full_like t u =
     '[ '("fill_value", AttrReq Double), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                    "int64", "int8", "uint8"]))),
        '("a", AttrOpt (t u))]

__npi_full_like ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_full_like t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__npi_full_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_full_like t u))
              args
        scalarArgs
          = catMaybes
              [("fill_value",) . showValue <$>
                 Just (Anon.get #fill_value fullArgs),
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_full_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_gamma t u =
     '[ '("shape", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_gamma ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_gamma t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_gamma t u)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Just (Anon.get #shape fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_greater t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_greater ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_greater t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t Bool)
__npi_greater args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_greater t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_greater_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_greater_equal ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_greater_equal t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t Bool)
__npi_greater_equal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_greater_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_greater_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_greater_equal_scalar ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList__npi_greater_equal_scalar t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t Bool)
__npi_greater_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_greater_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_greater_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_greater_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_greater_scalar ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__npi_greater_scalar t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t Bool)
__npi_greater_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_greater_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_greater_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_gumbel t u =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_gumbel ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_gumbel t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_gumbel args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_gumbel t u))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_gumbel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_hamming =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_hamming ::
              forall t u r .
                (Tensor t, FieldsAcc ParameterList__npi_hamming r, HasCallStack,
                 DType u) =>
                Record r -> TensorApply (t u)
__npi_hamming args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_hamming)) args
        scalarArgs
          = catMaybes
              [("m",) . showValue <$> Anon.get #m fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hamming" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_hanning =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_hanning ::
              forall t u r .
                (Tensor t, FieldsAcc ParameterList__npi_hanning r, HasCallStack,
                 DType u) =>
                Record r -> TensorApply (t u)
__npi_hanning args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_hanning)) args
        scalarArgs
          = catMaybes
              [("m",) . showValue <$> Anon.get #m fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hanning" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_hsplit t u =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("_data", AttrReq (t u))]

__npi_hsplit ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_hsplit t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_hsplit args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_hsplit t u))
              args
        scalarArgs
          = catMaybes
              [("indices",) . showValue <$> Just (Anon.get #indices fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("squeeze_axis",) . showValue <$> Anon.get #squeeze_axis fullArgs,
               ("sections",) . showValue <$> Anon.get #sections fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_hsplit" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_hsplit_backward = '[]

__npi_hsplit_backward ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__npi_hsplit_backward r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__npi_hsplit_backward args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_hsplit_backward))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hsplit_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_hstack t u =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("_data", AttrReq [t u])]

__npi_hstack ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_hstack t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_hstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_hstack t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("dim",) . showValue <$> Anon.get #dim fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_hstack" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_hypot t u =
     '[ '("x1", AttrOpt (t u)), '("x2", AttrOpt (t u))]

__npi_hypot ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_hypot t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_hypot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_hypot t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) . toRaw <$> Anon.get #x1 fullArgs,
               ("x2",) . toRaw <$> Anon.get #x2 fullArgs]
      in
      applyRaw "_npi_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_identity =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_identity ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__npi_identity r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__npi_identity args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_identity)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_identity" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_indices =
     '[ '("dimensions", AttrReq [Int]),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("ctx", AttrOpt Text)]

__npi_indices ::
              forall t u r .
                (Tensor t, FieldsAcc ParameterList__npi_indices r, HasCallStack,
                 DType u) =>
                Record r -> TensorApply (t u)
__npi_indices args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_indices)) args
        scalarArgs
          = catMaybes
              [("dimensions",) . showValue <$>
                 Just (Anon.get #dimensions fullArgs),
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_indices" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_insert_scalar t u =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u))]

__npi_insert_scalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_insert_scalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_insert_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_insert_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("val",) . showValue <$> Anon.get #val fullArgs,
               ("start",) . showValue <$> Anon.get #start fullArgs,
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("int_ind",) . showValue <$> Anon.get #int_ind fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> Anon.get #arr fullArgs,
               ("values",) . toRaw <$> Anon.get #values fullArgs]
      in
      applyRaw "_npi_insert_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_insert_slice t u =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u))]

__npi_insert_slice ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_insert_slice t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_insert_slice args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_insert_slice t u))
              args
        scalarArgs
          = catMaybes
              [("val",) . showValue <$> Anon.get #val fullArgs,
               ("start",) . showValue <$> Anon.get #start fullArgs,
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("int_ind",) . showValue <$> Anon.get #int_ind fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> Anon.get #arr fullArgs,
               ("values",) . toRaw <$> Anon.get #values fullArgs]
      in
      applyRaw "_npi_insert_slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_insert_tensor t u =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u)), '("obj", AttrOpt (t u))]

__npi_insert_tensor ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_insert_tensor t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_insert_tensor args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_insert_tensor t u))
              args
        scalarArgs
          = catMaybes
              [("val",) . showValue <$> Anon.get #val fullArgs,
               ("start",) . showValue <$> Anon.get #start fullArgs,
               ("stop",) . showValue <$> Anon.get #stop fullArgs,
               ("step",) . showValue <$> Anon.get #step fullArgs,
               ("int_ind",) . showValue <$> Anon.get #int_ind fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> Anon.get #arr fullArgs,
               ("values",) . toRaw <$> Anon.get #values fullArgs,
               ("obj",) . toRaw <$> Anon.get #obj fullArgs]
      in
      applyRaw "_npi_insert_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_isfinite t u = '[ '("x", AttrOpt (t u))]

__npi_isfinite ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_isfinite t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_isfinite args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_isfinite t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_isfinite" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_isinf t u = '[ '("x", AttrOpt (t u))]

__npi_isinf ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_isinf t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_isinf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_isinf t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_isinf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_isnan t u = '[ '("x", AttrOpt (t u))]

__npi_isnan ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_isnan t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_isnan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_isnan t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_isnan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_isneginf t u = '[ '("x", AttrOpt (t u))]

__npi_isneginf ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_isneginf t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_isneginf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_isneginf t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_isneginf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_isposinf t u = '[ '("x", AttrOpt (t u))]

__npi_isposinf ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_isposinf t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_isposinf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_isposinf t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_isposinf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_lcm t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_lcm ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_lcm t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_lcm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_lcm t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_lcm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_lcm_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_lcm_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_lcm_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_lcm_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_lcm_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_lcm_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_ldexp t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_ldexp ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_ldexp t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_ldexp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_ldexp t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_ldexp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_ldexp_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_ldexp_scalar ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_ldexp_scalar t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_ldexp_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_ldexp_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_ldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_less t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_less ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_less t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t Bool)
__npi_less args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_less t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_less" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_less_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_less_equal ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_less_equal t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t Bool)
__npi_less_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_less_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_less_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_less_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_less_equal_scalar ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__npi_less_equal_scalar t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t Bool)
__npi_less_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_less_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_less_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_less_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_less_scalar ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_less_scalar t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t Bool)
__npi_less_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_less_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_less_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_log t u = '[ '("x", AttrOpt (t u))]

__npi_log ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_log t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_log args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_log t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_log10 t u = '[ '("x", AttrOpt (t u))]

__npi_log10 ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_log10 t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_log10 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_log10 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_log1p t u = '[ '("x", AttrOpt (t u))]

__npi_log1p ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_log1p t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_log1p args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_log1p t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_log2 t u = '[ '("x", AttrOpt (t u))]

__npi_log2 ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_log2 t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_log2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_log2 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_logical_not t u = '[ '("x", AttrOpt (t u))]

__npi_logical_not ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_logical_not t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_logical_not args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_logical_not t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_logical_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_logistic t u =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_logistic ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_logistic t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_logistic args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_logistic t u))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_logistic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_logspace =
     '[ '("start", AttrReq Double), '("stop", AttrReq Double),
        '("num", AttrReq Int), '("endpoint", AttrOpt Bool),
        '("base", AttrOpt Double), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_logspace ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__npi_logspace r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__npi_logspace args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_logspace)) args
        scalarArgs
          = catMaybes
              [("start",) . showValue <$> Just (Anon.get #start fullArgs),
               ("stop",) . showValue <$> Just (Anon.get #stop fullArgs),
               ("num",) . showValue <$> Just (Anon.get #num fullArgs),
               ("endpoint",) . showValue <$> Anon.get #endpoint fullArgs,
               ("base",) . showValue <$> Anon.get #base fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_logspace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_matmul t u =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_matmul ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_matmul t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_matmul args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_matmul t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_matmul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_mean t u =
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
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_mean t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_mean args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_mean t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("initial",) . showValue <$> Anon.get #initial fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_mod t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_mod ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_mod t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_mod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_mod t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_mod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_mod_scalar ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_mod_scalar t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_mod_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_mod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_multinomial t u =
     '[ '("n", AttrReq Int), '("pvals", AttrOpt Int),
        '("size", AttrOpt (Maybe [Int])), '("a", AttrOpt (t u))]

__npi_multinomial ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_multinomial t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_multinomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_multinomial t u))
              args
        scalarArgs
          = catMaybes
              [("n",) . showValue <$> Just (Anon.get #n fullArgs),
               ("pvals",) . showValue <$> Anon.get #pvals fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_multiply t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_multiply ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_multiply t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_multiply args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_multiply t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_multiply" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_multiply_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_multiply_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__npi_multiply_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__npi_multiply_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_multiply_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_multiply_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_nan_to_num t u =
     '[ '("copy", AttrOpt Bool), '("nan", AttrOpt Double),
        '("posinf", AttrOpt (Maybe Double)),
        '("neginf", AttrOpt (Maybe Double)), '("_data", AttrReq (t u))]

__npi_nan_to_num ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_nan_to_num t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_nan_to_num args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_nan_to_num t u))
              args
        scalarArgs
          = catMaybes
              [("copy",) . showValue <$> Anon.get #copy fullArgs,
               ("nan",) . showValue <$> Anon.get #nan fullArgs,
               ("posinf",) . showValue <$> Anon.get #posinf fullArgs,
               ("neginf",) . showValue <$> Anon.get #neginf fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_nan_to_num" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_negative t u = '[ '("x", AttrOpt (t u))]

__npi_negative ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_negative t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_negative args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_negative t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_negative" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_norm t u = '[ '("_data", AttrReq (t u))]

__npi_norm ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_norm t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_norm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_norm t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_normal t u =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_normal ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_normal t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_normal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_normal t u))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_normal_n t u =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_normal_n ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_normal_n t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_normal_n args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_normal_n t u))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Just (Anon.get #loc fullArgs),
               ("scale",) . showValue <$> Just (Anon.get #scale fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_normal_n" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_not_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_not_equal ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_not_equal t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t Bool)
__npi_not_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_not_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_not_equal_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_not_equal_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__npi_not_equal_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t Bool)
__npi_not_equal_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_not_equal_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_not_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_ones =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_ones ::
           forall t u r .
             (Tensor t, FieldsAcc ParameterList__npi_ones r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_ones args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_ones)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_ones" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_pareto t u =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text), '("input1", AttrOpt (t u))]

__npi_pareto ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_pareto t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_pareto args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_pareto t u))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_pareto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_percentile t u =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("interpolation",
          AttrOpt
            (EnumType '["higher", "linear", "lower", "midpoint", "nearest"])),
        '("keepdims", AttrOpt Bool), '("q_scalar", AttrOpt (Maybe Double)),
        '("a", AttrOpt (t u)), '("q", AttrOpt (t u))]

__npi_percentile ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_percentile t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_percentile args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_percentile t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("interpolation",) . showValue <$>
                 Anon.get #interpolation fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("q_scalar",) . showValue <$> Anon.get #q_scalar fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("q",) . toRaw <$> Anon.get #q fullArgs]
      in
      applyRaw "_npi_percentile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_pinv t u =
     '[ '("hermitian", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("rcond", AttrOpt (t u))]

__npi_pinv ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_pinv t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_pinv args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_pinv t u)) args
        scalarArgs
          = catMaybes
              [("hermitian",) . showValue <$> Anon.get #hermitian fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("rcond",) . toRaw <$> Anon.get #rcond fullArgs]
      in
      applyRaw "_npi_pinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_pinv_scalar_rcond t u =
     '[ '("rcond", AttrOpt Double), '("hermitian", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_pinv_scalar_rcond ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__npi_pinv_scalar_rcond t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__npi_pinv_scalar_rcond args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_pinv_scalar_rcond t u))
              args
        scalarArgs
          = catMaybes
              [("rcond",) . showValue <$> Anon.get #rcond fullArgs,
               ("hermitian",) . showValue <$> Anon.get #hermitian fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_pinv_scalar_rcond" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_polyval t u =
     '[ '("p", AttrOpt (t u)), '("x", AttrOpt (t u))]

__npi_polyval ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_polyval t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_polyval args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_polyval t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("p",) . toRaw <$> Anon.get #p fullArgs,
               ("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_polyval" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_power t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_power ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_power t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_power args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_power t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_power_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_power_scalar ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_power_scalar t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_power_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_power_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_powerd t u =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("input1", AttrOpt (t u))]

__npi_powerd ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_powerd t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_powerd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_powerd t u))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_powerd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_radians t u = '[ '("x", AttrOpt (t u))]

__npi_radians ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_radians t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_radians args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_radians t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rarctan2_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rarctan2_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__npi_rarctan2_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__npi_rarctan2_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rarctan2_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rarctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rayleigh t u =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u))]

__npi_rayleigh ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_rayleigh t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_rayleigh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_rayleigh t u))
              args
        scalarArgs
          = catMaybes
              [("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_rayleigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rcopysign_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rcopysign_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__npi_rcopysign_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__npi_rcopysign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rcopysign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rcopysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_reciprocal t u = '[ '("x", AttrOpt (t u))]

__npi_reciprocal ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__npi_reciprocal t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__npi_reciprocal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_reciprocal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rint t u = '[ '("x", AttrOpt (t u))]

__npi_rint ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_rint t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_rint args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_rint t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_rint" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rldexp_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rldexp_scalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_rldexp_scalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_rldexp_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rldexp_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rmod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rmod_scalar ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_rmod_scalar t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_rmod_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rmod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rot90 t u =
     '[ '("k", AttrOpt Int), '("axes", AttrOpt (Maybe [Int])),
        '("_data", AttrReq (t u))]

__npi_rot90 ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_rot90 t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_rot90 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_rot90 t u)) args
        scalarArgs
          = catMaybes
              [("k",) . showValue <$> Anon.get #k fullArgs,
               ("axes",) . showValue <$> Anon.get #axes fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rot90" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rpower_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rpower_scalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_rpower_scalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_rpower_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rpower_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rsubtract_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rsubtract_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__npi_rsubtract_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__npi_rsubtract_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rsubtract_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rsubtract_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_rtrue_divide_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_rtrue_divide_scalar ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__npi_rtrue_divide_scalar t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__npi_rtrue_divide_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_rtrue_divide_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_rtrue_divide_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_share_memory t u =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_share_memory ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__npi_share_memory t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__npi_share_memory args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_share_memory t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_share_memory" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_sign t u = '[ '("x", AttrOpt (t u))]

__npi_sign ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_sign t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_sign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_sign t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_sin t u = '[ '("x", AttrOpt (t u))]

__npi_sin ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_sin t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_sin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_sin t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_sinh t u = '[ '("x", AttrOpt (t u))]

__npi_sinh ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_sinh t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_sinh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_sinh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_solve t u =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_solve ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_solve t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_solve args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_solve t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_solve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_sqrt t u = '[ '("x", AttrOpt (t u))]

__npi_sqrt ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_sqrt t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_sqrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_sqrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_square t u = '[ '("x", AttrOpt (t u))]

__npi_square ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_square t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_square args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_square t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_stack t u =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("_data", AttrReq [t u])]

__npi_stack ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_stack t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_stack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_stack t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_stack" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_std t u =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_std ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_std t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_std args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_std t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("ddof",) . showValue <$> Anon.get #ddof fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_std" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_subtract t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_subtract ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__npi_subtract t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__npi_subtract args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_subtract t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_subtract" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_subtract_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_subtract_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__npi_subtract_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__npi_subtract_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_subtract_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_subtract_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_svd t u = '[ '("a", AttrOpt (t u))]

__npi_svd ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_svd t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_svd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_svd t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_svd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tan t u = '[ '("x", AttrOpt (t u))]

__npi_tan ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_tan t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_tan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_tan t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tanh t u = '[ '("x", AttrOpt (t u))]

__npi_tanh ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_tanh t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_tanh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_tanh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tensordot t u =
     '[ '("a_axes_summed", AttrReq [Int]),
        '("b_axes_summed", AttrReq [Int]), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensordot ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_tensordot t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__npi_tensordot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_tensordot t u))
              args
        scalarArgs
          = catMaybes
              [("a_axes_summed",) . showValue <$>
                 Just (Anon.get #a_axes_summed fullArgs),
               ("b_axes_summed",) . showValue <$>
                 Just (Anon.get #b_axes_summed fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_tensordot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tensordot_int_axes t u =
     '[ '("axes", AttrReq Int), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensordot_int_axes ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__npi_tensordot_int_axes t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__npi_tensordot_int_axes args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_tensordot_int_axes t u))
              args
        scalarArgs
          = catMaybes
              [("axes",) . showValue <$> Just (Anon.get #axes fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_tensordot_int_axes" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tensorinv t u =
     '[ '("ind", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_tensorinv ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_tensorinv t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__npi_tensorinv args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_tensorinv t u))
              args
        scalarArgs
          = catMaybes [("ind",) . showValue <$> Anon.get #ind fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_tensorinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tensorsolve t u =
     '[ '("a_axes", AttrOpt [Int]), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensorsolve ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_tensorsolve t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_tensorsolve args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_tensorsolve t u))
              args
        scalarArgs
          = catMaybes [("a_axes",) . showValue <$> Anon.get #a_axes fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("b",) . toRaw <$> Anon.get #b fullArgs]
      in
      applyRaw "_npi_tensorsolve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_tril t u =
     '[ '("k", AttrOpt Int), '("_data", AttrReq (t u))]

__npi_tril ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npi_tril t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npi_tril args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_tril t u)) args
        scalarArgs
          = catMaybes [("k",) . showValue <$> Anon.get #k fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_tril" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_true_divide t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_true_divide ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList__npi_true_divide t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
__npi_true_divide args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_true_divide t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_npi_true_divide" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_true_divide_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__npi_true_divide_scalar ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__npi_true_divide_scalar t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__npi_true_divide_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_true_divide_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_true_divide_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_trunc t u = '[ '("x", AttrOpt (t u))]

__npi_trunc ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_trunc t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_trunc args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_trunc t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_trunc" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_uniform t u =
     '[ '("low", AttrReq (Maybe Float)),
        '("high", AttrReq (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_uniform ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_uniform t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_uniform args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_uniform t u))
              args
        scalarArgs
          = catMaybes
              [("low",) . showValue <$> Just (Anon.get #low fullArgs),
               ("high",) . showValue <$> Just (Anon.get #high fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_uniform_n t u =
     '[ '("low", AttrReq (Maybe Float)),
        '("high", AttrReq (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_uniform_n ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__npi_uniform_n t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__npi_uniform_n args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_uniform_n t u))
              args
        scalarArgs
          = catMaybes
              [("low",) . showValue <$> Just (Anon.get #low fullArgs),
               ("high",) . showValue <$> Just (Anon.get #high fullArgs),
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> Anon.get #input1 fullArgs,
               ("input2",) . toRaw <$> Anon.get #input2 fullArgs]
      in
      applyRaw "_npi_uniform_n" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_unique t u =
     '[ '("return_index", AttrOpt Bool),
        '("return_inverse", AttrOpt Bool),
        '("return_counts", AttrOpt Bool), '("axis", AttrOpt (Maybe Int)),
        '("_data", AttrReq (t u))]

__npi_unique ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_unique t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_unique args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_unique t u))
              args
        scalarArgs
          = catMaybes
              [("return_index",) . showValue <$> Anon.get #return_index fullArgs,
               ("return_inverse",) . showValue <$>
                 Anon.get #return_inverse fullArgs,
               ("return_counts",) . showValue <$>
                 Anon.get #return_counts fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npi_unique" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_var t u =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_var ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__npi_var t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__npi_var args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_var t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("ddof",) . showValue <$> Anon.get #ddof fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npi_var" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_vstack t u =
     '[ '("num_args", AttrReq Int), '("_data", AttrReq [t u])]

__npi_vstack ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__npi_vstack t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__npi_vstack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_vstack t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_npi_vstack" scalarArgs (Right tensorVarArgs)

type ParameterList__npi_weibull t u =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text), '("input1", AttrOpt (t u))]

__npi_weibull ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npi_weibull t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npi_weibull args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_weibull t u))
              args
        scalarArgs
          = catMaybes
              [("a",) . showValue <$> Anon.get #a fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs]
        tensorKeyArgs
          = catMaybes [("input1",) . toRaw <$> Anon.get #input1 fullArgs]
      in
      applyRaw "_npi_weibull" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_where t u =
     '[ '("condition", AttrOpt (t Bool)), '("x", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

__npi_where ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList__npi_where t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
__npi_where args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_where t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> Anon.get #condition fullArgs,
               ("x",) . toRaw <$> Anon.get #x fullArgs,
               ("y",) . toRaw <$> Anon.get #y fullArgs]
      in
      applyRaw "_npi_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_where_lscalar t u =
     '[ '("scalar", AttrOpt Double), '("condition", AttrOpt (t u)),
        '("x", AttrOpt (t u))]

__npi_where_lscalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_where_lscalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_where_lscalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_where_lscalar t u))
              args
        scalarArgs
          = catMaybes [("scalar",) . showValue <$> Anon.get #scalar fullArgs]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> Anon.get #condition fullArgs,
               ("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npi_where_lscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_where_rscalar t u =
     '[ '("scalar", AttrOpt Double), '("condition", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

__npi_where_rscalar ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_where_rscalar t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_where_rscalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_where_rscalar t u))
              args
        scalarArgs
          = catMaybes [("scalar",) . showValue <$> Anon.get #scalar fullArgs]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> Anon.get #condition fullArgs,
               ("y",) . toRaw <$> Anon.get #y fullArgs]
      in
      applyRaw "_npi_where_rscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_where_scalar2 t u =
     '[ '("x", AttrOpt Double), '("y", AttrOpt Double),
        '("condition", AttrOpt (t u))]

__npi_where_scalar2 ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__npi_where_scalar2 t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__npi_where_scalar2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npi_where_scalar2 t u))
              args
        scalarArgs
          = catMaybes
              [("x",) . showValue <$> Anon.get #x fullArgs,
               ("y",) . showValue <$> Anon.get #y fullArgs]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> Anon.get #condition fullArgs]
      in
      applyRaw "_npi_where_scalar2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npi_zeros =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_zeros ::
            forall t u r .
              (Tensor t, FieldsAcc ParameterList__npi_zeros r, HasCallStack,
               DType u) =>
              Record r -> TensorApply (t u)
__npi_zeros args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npi_zeros)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_zeros" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npx_constraint_check t u =
     '[ '("msg", AttrOpt Text), '("input", AttrOpt (t u))]

__npx_constraint_check ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__npx_constraint_check t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__npx_constraint_check args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__npx_constraint_check t u))
              args
        scalarArgs
          = catMaybes [("msg",) . showValue <$> Anon.get #msg fullArgs]
        tensorKeyArgs
          = catMaybes [("input",) . toRaw <$> Anon.get #input fullArgs]
      in
      applyRaw "_npx_constraint_check" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npx_nonzero t u = '[ '("x", AttrOpt (t u))]

__npx_nonzero ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npx_nonzero t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npx_nonzero args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npx_nonzero t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) . toRaw <$> Anon.get #x fullArgs]
      in
      applyRaw "_npx_nonzero" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npx_relu t u = '[ '("_data", AttrReq (t u))]

__npx_relu ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__npx_relu t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__npx_relu args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npx_relu t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npx_relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npx_reshape t u =
     '[ '("newshape", AttrReq [Int]), '("reverse", AttrOpt Bool),
        '("order", AttrOpt Text), '("a", AttrOpt (t u))]

__npx_reshape ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npx_reshape t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npx_reshape args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npx_reshape t u))
              args
        scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> Just (Anon.get #newshape fullArgs),
               ("reverse",) . showValue <$> Anon.get #reverse fullArgs,
               ("order",) . showValue <$> Anon.get #order fullArgs]
        tensorKeyArgs = catMaybes [("a",) . toRaw <$> Anon.get #a fullArgs]
      in
      applyRaw "_npx_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__npx_sigmoid t u = '[ '("_data", AttrReq (t u))]

__npx_sigmoid ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__npx_sigmoid t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__npx_sigmoid args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__npx_sigmoid t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_npx_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__onehot_encode t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__onehot_encode ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__onehot_encode t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__onehot_encode args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__onehot_encode t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_onehot_encode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__ones =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__ones ::
       forall t u r .
         (Tensor t, FieldsAcc ParameterList__ones r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
__ones args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__ones)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_ones" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__plus_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__plus_scalar ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__plus_scalar t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__plus_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__plus_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_plus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__power t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__power ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList__power t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
__power args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__power t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__power_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__power_scalar ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__power_scalar t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__power_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__power_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_exponential =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_exponential ::
                     forall t u r .
                       (Tensor t, FieldsAcc ParameterList__random_exponential r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__random_exponential args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__random_exponential))
              args
        scalarArgs
          = catMaybes
              [("lam",) . showValue <$> Anon.get #lam fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_exponential_like t u =
     '[ '("lam", AttrOpt Float), '("_data", AttrReq (t u))]

__random_exponential_like ::
                          forall t u r .
                            (Tensor t,
                             FieldsAcc (ParameterList__random_exponential_like t u) r,
                             HasCallStack, DType u) =>
                            Record r -> TensorApply (t u)
__random_exponential_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_exponential_like t u))
              args
        scalarArgs
          = catMaybes [("lam",) . showValue <$> Anon.get #lam fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_exponential_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_gamma =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_gamma ::
               forall t u r .
                 (Tensor t, FieldsAcc ParameterList__random_gamma r, HasCallStack,
                  DType u) =>
                 Record r -> TensorApply (t u)
__random_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__random_gamma)) args
        scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_gamma_like t u =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("_data", AttrReq (t u))]

__random_gamma_like ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__random_gamma_like t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__random_gamma_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_gamma_like t u))
              args
        scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_gamma_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_generalized_negative_binomial =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_generalized_negative_binomial ::
                                       forall t u r .
                                         (Tensor t,
                                          FieldsAcc
                                            ParameterList__random_generalized_negative_binomial
                                            r,
                                          HasCallStack, DType u) =>
                                         Record r -> TensorApply (t u)
__random_generalized_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_generalized_negative_binomial))
              args
        scalarArgs
          = catMaybes
              [("mu",) . showValue <$> Anon.get #mu fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_generalized_negative_binomial_like t u =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("_data", AttrReq (t u))]

__random_generalized_negative_binomial_like ::
                                            forall t u r .
                                              (Tensor t,
                                               FieldsAcc
                                                 (ParameterList__random_generalized_negative_binomial_like
                                                    t
                                                    u)
                                                 r,
                                               HasCallStack, DType u) =>
                                              Record r -> TensorApply (t u)
__random_generalized_negative_binomial_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__random_generalized_negative_binomial_like t u))
              args
        scalarArgs
          = catMaybes
              [("mu",) . showValue <$> Anon.get #mu fullArgs,
               ("alpha",) . showValue <$> Anon.get #alpha fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_generalized_negative_binomial_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_negative_binomial =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_negative_binomial ::
                           forall t u r .
                             (Tensor t, FieldsAcc ParameterList__random_negative_binomial r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__random_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_negative_binomial))
              args
        scalarArgs
          = catMaybes
              [("k",) . showValue <$> Anon.get #k fullArgs,
               ("p",) . showValue <$> Anon.get #p fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_negative_binomial_like t u =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("_data", AttrReq (t u))]

__random_negative_binomial_like ::
                                forall t u r .
                                  (Tensor t,
                                   FieldsAcc (ParameterList__random_negative_binomial_like t u) r,
                                   HasCallStack, DType u) =>
                                  Record r -> TensorApply (t u)
__random_negative_binomial_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_negative_binomial_like t u))
              args
        scalarArgs
          = catMaybes
              [("k",) . showValue <$> Anon.get #k fullArgs,
               ("p",) . showValue <$> Anon.get #p fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_negative_binomial_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_normal =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_normal ::
                forall t u r .
                  (Tensor t, FieldsAcc ParameterList__random_normal r, HasCallStack,
                   DType u) =>
                  Record r -> TensorApply (t u)
__random_normal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__random_normal)) args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Anon.get #loc fullArgs,
               ("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_normal_like t u =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("_data", AttrReq (t u))]

__random_normal_like ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__random_normal_like t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__random_normal_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_normal_like t u))
              args
        scalarArgs
          = catMaybes
              [("loc",) . showValue <$> Anon.get #loc fullArgs,
               ("scale",) . showValue <$> Anon.get #scale fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_normal_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_dirichlet t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("alpha", AttrOpt (t u))]

__random_pdf_dirichlet ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__random_pdf_dirichlet t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__random_pdf_dirichlet args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_dirichlet t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("alpha",) . toRaw <$> Anon.get #alpha fullArgs]
      in
      applyRaw "_random_pdf_dirichlet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_exponential t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("lam", AttrOpt (t u))]

__random_pdf_exponential ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList__random_pdf_exponential t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
__random_pdf_exponential args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_exponential t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("lam",) . toRaw <$> Anon.get #lam fullArgs]
      in
      applyRaw "_random_pdf_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_gamma t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("alpha", AttrOpt (t u)), '("beta", AttrOpt (t u))]

__random_pdf_gamma ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__random_pdf_gamma t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__random_pdf_gamma args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_gamma t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("alpha",) . toRaw <$> Anon.get #alpha fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "_random_pdf_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_generalized_negative_binomial t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("mu", AttrOpt (t u)), '("alpha", AttrOpt (t u))]

__random_pdf_generalized_negative_binomial ::
                                           forall t u r .
                                             (Tensor t,
                                              FieldsAcc
                                                (ParameterList__random_pdf_generalized_negative_binomial
                                                   t
                                                   u)
                                                r,
                                              HasCallStack, DType u) =>
                                             Record r -> TensorApply (t u)
__random_pdf_generalized_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy
                 @(ParameterList__random_pdf_generalized_negative_binomial t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("mu",) . toRaw <$> Anon.get #mu fullArgs,
               ("alpha",) . toRaw <$> Anon.get #alpha fullArgs]
      in
      applyRaw "_random_pdf_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_negative_binomial t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("k", AttrOpt (t u)), '("p", AttrOpt (t u))]

__random_pdf_negative_binomial ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList__random_pdf_negative_binomial t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
__random_pdf_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_negative_binomial t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("k",) . toRaw <$> Anon.get #k fullArgs,
               ("p",) . toRaw <$> Anon.get #p fullArgs]
      in
      applyRaw "_random_pdf_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_normal t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("mu", AttrOpt (t u)), '("sigma", AttrOpt (t u))]

__random_pdf_normal ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__random_pdf_normal t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__random_pdf_normal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_normal t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("mu",) . toRaw <$> Anon.get #mu fullArgs,
               ("sigma",) . toRaw <$> Anon.get #sigma fullArgs]
      in
      applyRaw "_random_pdf_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_poisson t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("lam", AttrOpt (t u))]

__random_pdf_poisson ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__random_pdf_poisson t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__random_pdf_poisson args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_poisson t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("lam",) . toRaw <$> Anon.get #lam fullArgs]
      in
      applyRaw "_random_pdf_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_pdf_uniform t u =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("low", AttrOpt (t u)), '("high", AttrOpt (t u))]

__random_pdf_uniform ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__random_pdf_uniform t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__random_pdf_uniform args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_pdf_uniform t u))
              args
        scalarArgs
          = catMaybes [("is_log",) . showValue <$> Anon.get #is_log fullArgs]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> Anon.get #sample fullArgs,
               ("low",) . toRaw <$> Anon.get #low fullArgs,
               ("high",) . toRaw <$> Anon.get #high fullArgs]
      in
      applyRaw "_random_pdf_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_poisson =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_poisson ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__random_poisson r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__random_poisson args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__random_poisson))
              args
        scalarArgs
          = catMaybes
              [("lam",) . showValue <$> Anon.get #lam fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_poisson_like t u =
     '[ '("lam", AttrOpt Float), '("_data", AttrReq (t u))]

__random_poisson_like ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__random_poisson_like t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__random_poisson_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_poisson_like t u))
              args
        scalarArgs
          = catMaybes [("lam",) . showValue <$> Anon.get #lam fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_poisson_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_uniform =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_uniform ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__random_uniform r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__random_uniform args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__random_uniform))
              args
        scalarArgs
          = catMaybes
              [("low",) . showValue <$> Anon.get #low fullArgs,
               ("high",) . showValue <$> Anon.get #high fullArgs,
               ("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__random_uniform_like t u =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("_data", AttrReq (t u))]

__random_uniform_like ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__random_uniform_like t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__random_uniform_like args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__random_uniform_like t u))
              args
        scalarArgs
          = catMaybes
              [("low",) . showValue <$> Anon.get #low fullArgs,
               ("high",) . showValue <$> Anon.get #high fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_random_uniform_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__ravel_multi_index t u =
     '[ '("shape", AttrOpt [Int]), '("_data", AttrReq (t u))]

__ravel_multi_index ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList__ravel_multi_index t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__ravel_multi_index args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__ravel_multi_index t u))
              args
        scalarArgs
          = catMaybes [("shape",) . showValue <$> Anon.get #shape fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_ravel_multi_index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__rdiv_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__rdiv_scalar ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__rdiv_scalar t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__rdiv_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__rdiv_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_rdiv_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__rminus_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__rminus_scalar ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__rminus_scalar t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__rminus_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__rminus_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_rminus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__rmod_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__rmod_scalar ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList__rmod_scalar t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
__rmod_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__rmod_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__rnn_param_concat t u =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("_data", AttrReq [t u])]

__rnn_param_concat ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList__rnn_param_concat t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
__rnn_param_concat args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__rnn_param_concat t u))
              args
        scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs),
               ("dim",) . showValue <$> Anon.get #dim fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "_rnn_param_concat" scalarArgs (Right tensorVarArgs)

type ParameterList__rpower_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__rpower_scalar ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__rpower_scalar t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__rpower_scalar args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__rpower_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_exponential t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt (t u))]

__sample_exponential ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__sample_exponential t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__sample_exponential args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sample_exponential t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("lam",) . toRaw <$> Anon.get #lam fullArgs]
      in
      applyRaw "_sample_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_gamma t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("alpha", AttrOpt (t u)), '("beta", AttrOpt (t u))]

__sample_gamma ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__sample_gamma t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__sample_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sample_gamma t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("alpha",) . toRaw <$> Anon.get #alpha fullArgs,
               ("beta",) . toRaw <$> Anon.get #beta fullArgs]
      in
      applyRaw "_sample_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_generalized_negative_binomial t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt (t u)), '("alpha", AttrOpt (t u))]

__sample_generalized_negative_binomial ::
                                       forall t u r .
                                         (Tensor t,
                                          FieldsAcc
                                            (ParameterList__sample_generalized_negative_binomial t
                                               u)
                                            r,
                                          HasCallStack, DType u) =>
                                         Record r -> TensorApply (t u)
__sample_generalized_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sample_generalized_negative_binomial t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("mu",) . toRaw <$> Anon.get #mu fullArgs,
               ("alpha",) . toRaw <$> Anon.get #alpha fullArgs]
      in
      applyRaw "_sample_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_multinomial t u =
     '[ '("shape", AttrOpt [Int]), '("get_prob", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("_data", AttrReq (t u))]

__sample_multinomial ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList__sample_multinomial t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
__sample_multinomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sample_multinomial t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("get_prob",) . showValue <$> Anon.get #get_prob fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_sample_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_negative_binomial t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("k", AttrOpt (t u)), '("p", AttrOpt (t u))]

__sample_negative_binomial ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList__sample_negative_binomial t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__sample_negative_binomial args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sample_negative_binomial t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("k",) . toRaw <$> Anon.get #k fullArgs,
               ("p",) . toRaw <$> Anon.get #p fullArgs]
      in
      applyRaw "_sample_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_normal t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt (t u)), '("sigma", AttrOpt (t u))]

__sample_normal ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__sample_normal t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__sample_normal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sample_normal t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("mu",) . toRaw <$> Anon.get #mu fullArgs,
               ("sigma",) . toRaw <$> Anon.get #sigma fullArgs]
      in
      applyRaw "_sample_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_poisson t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt (t u))]

__sample_poisson ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__sample_poisson t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__sample_poisson args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sample_poisson t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("lam",) . toRaw <$> Anon.get #lam fullArgs]
      in
      applyRaw "_sample_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_uniform t u =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("low", AttrOpt (t u)), '("high", AttrOpt (t u))]

__sample_uniform ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__sample_uniform t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__sample_uniform args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sample_uniform t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("low",) . toRaw <$> Anon.get #low fullArgs,
               ("high",) . toRaw <$> Anon.get #high fullArgs]
      in
      applyRaw "_sample_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sample_unique_zipfian =
     '[ '("range_max", AttrReq Int), '("shape", AttrOpt [Int])]

__sample_unique_zipfian ::
                        forall t u r .
                          (Tensor t, FieldsAcc ParameterList__sample_unique_zipfian r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__sample_unique_zipfian args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sample_unique_zipfian))
              args
        scalarArgs
          = catMaybes
              [("range_max",) . showValue <$>
                 Just (Anon.get #range_max fullArgs),
               ("shape",) . showValue <$> Anon.get #shape fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sample_unique_zipfian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__scatter_elemwise_div t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__scatter_elemwise_div ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__scatter_elemwise_div t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__scatter_elemwise_div args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__scatter_elemwise_div t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_scatter_elemwise_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__scatter_minus_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__scatter_minus_scalar ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__scatter_minus_scalar t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__scatter_minus_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__scatter_minus_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_scatter_minus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__scatter_plus_scalar t u =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("_data", AttrReq (t u))]

__scatter_plus_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__scatter_plus_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__scatter_plus_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__scatter_plus_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("is_int",) . showValue <$> Anon.get #is_int fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_scatter_plus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__scatter_set_nd t u =
     '[ '("shape", AttrReq [Int]), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u)), '("indices", AttrOpt (t u))]

__scatter_set_nd ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList__scatter_set_nd t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
__scatter_set_nd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__scatter_set_nd t u))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Just (Anon.get #shape fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs,
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "_scatter_set_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__set_value = '[ '("src", AttrOpt Float)]

__set_value ::
            forall t u r .
              (Tensor t, FieldsAcc ParameterList__set_value r, HasCallStack,
               DType u) =>
              Record r -> TensorApply (t u)
__set_value args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__set_value)) args
        scalarArgs
          = catMaybes [("src",) . showValue <$> Anon.get #src fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_set_value" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sg_mkldnn_conv = '[]

__sg_mkldnn_conv ::
                 forall t u r .
                   (Tensor t, FieldsAcc ParameterList__sg_mkldnn_conv r, HasCallStack,
                    DType u) =>
                   Record r -> TensorApply (t u)
__sg_mkldnn_conv args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sg_mkldnn_conv))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sg_mkldnn_conv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sg_mkldnn_fully_connected = '[]

__sg_mkldnn_fully_connected ::
                            forall t u r .
                              (Tensor t, FieldsAcc ParameterList__sg_mkldnn_fully_connected r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
__sg_mkldnn_fully_connected args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sg_mkldnn_fully_connected))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sg_mkldnn_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sg_mkldnn_selfatt_qk t u =
     '[ '("heads", AttrReq Int), '("quantized", AttrOpt Bool),
        '("enable_float_output", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("queries_keys_values", AttrOpt (t u))]

__sg_mkldnn_selfatt_qk ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList__sg_mkldnn_selfatt_qk t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
__sg_mkldnn_selfatt_qk args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sg_mkldnn_selfatt_qk t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs),
               ("quantized",) . showValue <$> Anon.get #quantized fullArgs,
               ("enable_float_output",) . showValue <$>
                 Anon.get #enable_float_output fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 Anon.get #queries_keys_values fullArgs]
      in
      applyRaw "_sg_mkldnn_selfatt_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sg_mkldnn_selfatt_valatt t u =
     '[ '("heads", AttrReq Int), '("quantized", AttrOpt Bool),
        '("enable_float_output", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("queries_keys_values", AttrOpt (t u)),
        '("attention", AttrOpt (t u))]

__sg_mkldnn_selfatt_valatt ::
                           forall t u r .
                             (Tensor t,
                              FieldsAcc (ParameterList__sg_mkldnn_selfatt_valatt t u) r,
                              HasCallStack, DType u) =>
                             Record r -> TensorApply (t u)
__sg_mkldnn_selfatt_valatt args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sg_mkldnn_selfatt_valatt t u))
              args
        scalarArgs
          = catMaybes
              [("heads",) . showValue <$> Just (Anon.get #heads fullArgs),
               ("quantized",) . showValue <$> Anon.get #quantized fullArgs,
               ("enable_float_output",) . showValue <$>
                 Anon.get #enable_float_output fullArgs,
               ("min_calib_range",) . showValue <$>
                 Anon.get #min_calib_range fullArgs,
               ("max_calib_range",) . showValue <$>
                 Anon.get #max_calib_range fullArgs]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 Anon.get #queries_keys_values fullArgs,
               ("attention",) . toRaw <$> Anon.get #attention fullArgs]
      in
      applyRaw "_sg_mkldnn_selfatt_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__shuffle t u = '[ '("_data", AttrReq (t u))]

__shuffle ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList__shuffle t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
__shuffle args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__shuffle t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_shuffle" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__slice_assign t u =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

__slice_assign ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList__slice_assign t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
__slice_assign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__slice_assign t u))
              args
        scalarArgs
          = catMaybes
              [("begin",) . showValue <$> Just (Anon.get #begin fullArgs),
               ("end",) . showValue <$> Just (Anon.get #end fullArgs),
               ("step",) . showValue <$> Anon.get #step fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "_slice_assign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__slice_assign_scalar t u =
     '[ '("scalar", AttrOpt Double), '("begin", AttrReq [Int]),
        '("end", AttrReq [Int]), '("step", AttrOpt [Int]),
        '("_data", AttrReq (t u))]

__slice_assign_scalar ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList__slice_assign_scalar t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__slice_assign_scalar args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__slice_assign_scalar t u))
              args
        scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> Anon.get #scalar fullArgs,
               ("begin",) . showValue <$> Just (Anon.get #begin fullArgs),
               ("end",) . showValue <$> Just (Anon.get #end fullArgs),
               ("step",) . showValue <$> Anon.get #step fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_slice_assign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sparse_adagrad_update t u =
     '[ '("lr", AttrReq Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("history", AttrOpt (t u))]

__sparse_adagrad_update ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList__sparse_adagrad_update t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
__sparse_adagrad_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__sparse_adagrad_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("history",) . toRaw <$> Anon.get #history fullArgs]
      in
      applyRaw "_sparse_adagrad_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__sparse_retain t u =
     '[ '("_data", AttrReq (t u)), '("indices", AttrOpt (t u))]

__sparse_retain ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__sparse_retain t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__sparse_retain args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__sparse_retain t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "_sparse_retain" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__split_v2 t u =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("_data", AttrReq (t u))]

__split_v2 ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList__split_v2 t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
__split_v2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__split_v2 t u)) args
        scalarArgs
          = catMaybes
              [("indices",) . showValue <$> Just (Anon.get #indices fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("squeeze_axis",) . showValue <$> Anon.get #squeeze_axis fullArgs,
               ("sections",) . showValue <$> Anon.get #sections fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_split_v2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__split_v2_backward = '[]

__split_v2_backward ::
                    forall t u r .
                      (Tensor t, FieldsAcc ParameterList__split_v2_backward r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
__split_v2_backward args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__split_v2_backward))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_split_v2_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__square_sum t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

__square_sum ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList__square_sum t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
__square_sum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__square_sum t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_square_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__unravel_index t u =
     '[ '("shape", AttrOpt [Int]), '("_data", AttrReq (t u))]

__unravel_index ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList__unravel_index t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
__unravel_index args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__unravel_index t u))
              args
        scalarArgs
          = catMaybes [("shape",) . showValue <$> Anon.get #shape fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "_unravel_index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__zeros =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__zeros ::
        forall t u r .
          (Tensor t, FieldsAcc ParameterList__zeros r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
__zeros args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList__zeros)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_zeros" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList__zeros_without_dtype =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt Int)]

__zeros_without_dtype ::
                      forall t u r .
                        (Tensor t, FieldsAcc ParameterList__zeros_without_dtype r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
__zeros_without_dtype args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList__zeros_without_dtype))
              args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Anon.get #shape fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_zeros_without_dtype" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_abs t u = '[ '("_data", AttrReq (t u))]

_abs ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_abs t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_abs args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_abs t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "abs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_adam_update t u =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mean", AttrOpt (t u)), '("var", AttrOpt (t u))]

_adam_update ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_adam_update t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_adam_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_adam_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("lazy_update",) . showValue <$> Anon.get #lazy_update fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mean",) . toRaw <$> Anon.get #mean fullArgs,
               ("var",) . toRaw <$> Anon.get #var fullArgs]
      in
      applyRaw "adam_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_add_n t u = '[ '("args", AttrReq [t u])]

_add_n ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_add_n t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_add_n args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_add_n t u)) args
        scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #args fullArgs :: [RawTensor t]
      in applyRaw "add_n" scalarArgs (Right tensorVarArgs)

type ParameterList_all_finite t u =
     '[ '("init_output", AttrOpt Bool), '("_data", AttrReq (t u))]

_all_finite ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_all_finite t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_all_finite args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_all_finite t u)) args
        scalarArgs
          = catMaybes
              [("init_output",) . showValue <$> Anon.get #init_output fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "all_finite" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_amp_cast t u =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("_data", AttrReq (t u))]

_amp_cast ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList_amp_cast t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
_amp_cast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_amp_cast t u)) args
        scalarArgs
          = catMaybes
              [("dtype",) . showValue <$> Just (Anon.get #dtype fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "amp_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_amp_multicast t u =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("_data", AttrReq [t u])]

_amp_multicast ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_amp_multicast t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_amp_multicast args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_amp_multicast t u))
              args
        scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 Just (Anon.get #num_outputs fullArgs),
               ("cast_narrow",) . showValue <$> Anon.get #cast_narrow fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "amp_multicast" scalarArgs (Right tensorVarArgs)

type ParameterList_arccos t u = '[ '("_data", AttrReq (t u))]

_arccos ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_arccos t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_arccos args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arccos t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_arccosh t u = '[ '("_data", AttrReq (t u))]

_arccosh ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_arccosh t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_arccosh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arccosh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_arcsin t u = '[ '("_data", AttrReq (t u))]

_arcsin ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_arcsin t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_arcsin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arcsin t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_arcsinh t u = '[ '("_data", AttrReq (t u))]

_arcsinh ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_arcsinh t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_arcsinh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arcsinh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_arctan t u = '[ '("_data", AttrReq (t u))]

_arctan ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_arctan t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_arctan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arctan t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_arctanh t u = '[ '("_data", AttrReq (t u))]

_arctanh ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_arctanh t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_arctanh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_arctanh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_argmax t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

_argmax ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_argmax t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_argmax args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_argmax t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "argmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_argmax_channel t u =
     '[ '("_data", AttrReq (t u))]

_argmax_channel ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_argmax_channel t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_argmax_channel args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_argmax_channel t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "argmax_channel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_argmin t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

_argmin ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_argmin t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_argmin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_argmin t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "argmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_argsort t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("_data", AttrReq (t u))]

_argsort ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_argsort t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_argsort args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_argsort t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("is_ascend",) . showValue <$> Anon.get #is_ascend fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "argsort" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_batch_dot t u =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_batch_dot ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_batch_dot t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_batch_dot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_batch_dot t u)) args
        scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$> Anon.get #transpose_a fullArgs,
               ("transpose_b",) . showValue <$> Anon.get #transpose_b fullArgs,
               ("forward_stype",) . showValue <$>
                 Anon.get #forward_stype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "batch_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_batch_take t u =
     '[ '("a", AttrOpt (t u)), '("indices", AttrOpt (t u))]

_batch_take ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_batch_take t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_batch_take args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_batch_take t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "batch_take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_add t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_add ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_broadcast_add t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_broadcast_add args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_add t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_axis t u =
     '[ '("axis", AttrOpt [Int]), '("size", AttrOpt [Int]),
        '("_data", AttrReq (t u))]

_broadcast_axis ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_broadcast_axis t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_broadcast_axis args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_axis t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("size",) . showValue <$> Anon.get #size fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "broadcast_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_div t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_div ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_broadcast_div t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_broadcast_div args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_div t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_equal ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_broadcast_equal t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_broadcast_equal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_greater t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_greater ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_broadcast_greater t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_broadcast_greater args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_greater t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_greater_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_greater_equal ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList_broadcast_greater_equal t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
_broadcast_greater_equal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_greater_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_hypot t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_hypot ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_broadcast_hypot t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_broadcast_hypot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_hypot t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_lesser t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_lesser ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList_broadcast_lesser t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
_broadcast_lesser args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_lesser t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_lesser" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_lesser_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_lesser_equal ::
                        forall t u r .
                          (Tensor t, FieldsAcc (ParameterList_broadcast_lesser_equal t u) r,
                           HasCallStack, DType u) =>
                          Record r -> TensorApply (t u)
_broadcast_lesser_equal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_lesser_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_lesser_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_like t u =
     '[ '("lhs_axes", AttrOpt (Maybe [Int])),
        '("rhs_axes", AttrOpt (Maybe [Int])), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_broadcast_like ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_broadcast_like t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_broadcast_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_like t u))
              args
        scalarArgs
          = catMaybes
              [("lhs_axes",) . showValue <$> Anon.get #lhs_axes fullArgs,
               ("rhs_axes",) . showValue <$> Anon.get #rhs_axes fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_logical_and t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_and ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList_broadcast_logical_and t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
_broadcast_logical_and args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_logical_and t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_logical_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_logical_or t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_or ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList_broadcast_logical_or t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
_broadcast_logical_or args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_logical_or t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_logical_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_logical_xor t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_xor ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList_broadcast_logical_xor t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
_broadcast_logical_xor args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_logical_xor t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_logical_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_maximum t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_maximum ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_broadcast_maximum t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_broadcast_maximum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_maximum t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_minimum t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_minimum ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_broadcast_minimum t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_broadcast_minimum args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_minimum t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_mod t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_mod ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_broadcast_mod t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_broadcast_mod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_mod t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_mul t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_mul ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_broadcast_mul t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_broadcast_mul args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_mul t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_not_equal t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_not_equal ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList_broadcast_not_equal t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
_broadcast_not_equal args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_broadcast_not_equal t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_power t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_power ::
                 forall t u r .
                   (Tensor t, FieldsAcc (ParameterList_broadcast_power t u) r,
                    HasCallStack, DType u) =>
                   Record r -> TensorApply (t u)
_broadcast_power args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_power t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_sub t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_sub ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_broadcast_sub t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_broadcast_sub args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_sub t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_broadcast_to t u =
     '[ '("shape", AttrOpt [Int]), '("_data", AttrReq (t u))]

_broadcast_to ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_broadcast_to t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_broadcast_to args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_broadcast_to t u))
              args
        scalarArgs
          = catMaybes [("shape",) . showValue <$> Anon.get #shape fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_cast_storage t u =
     '[ '("stype",
          AttrReq (EnumType '["csr", "default", "row_sparse"])),
        '("_data", AttrReq (t u))]

_cast_storage ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_cast_storage t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_cast_storage args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_cast_storage t u))
              args
        scalarArgs
          = catMaybes
              [("stype",) . showValue <$> Just (Anon.get #stype fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "cast_storage" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_cbrt t u = '[ '("_data", AttrReq (t u))]

_cbrt ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_cbrt t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_cbrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_cbrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_ceil t u = '[ '("_data", AttrReq (t u))]

_ceil ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_ceil t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_ceil args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ceil t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "ceil" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_clip t u =
     '[ '("a_min", AttrReq Float), '("a_max", AttrReq Float),
        '("_data", AttrReq (t u))]

_clip ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_clip t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_clip args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_clip t u)) args
        scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> Just (Anon.get #a_min fullArgs),
               ("a_max",) . showValue <$> Just (Anon.get #a_max fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "clip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_col2im t u =
     '[ '("output_size", AttrReq [Int]), '("kernel", AttrReq [Int]),
        '("stride", AttrOpt [Int]), '("dilate", AttrOpt [Int]),
        '("pad", AttrOpt [Int]), '("_data", AttrReq (t u))]

_col2im ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_col2im t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_col2im args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_col2im t u)) args
        scalarArgs
          = catMaybes
              [("output_size",) . showValue <$>
                 Just (Anon.get #output_size fullArgs),
               ("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "col2im" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_cos t u = '[ '("_data", AttrReq (t u))]

_cos ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_cos t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_cos args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_cos t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_cosh t u = '[ '("_data", AttrReq (t u))]

_cosh ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_cosh t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_cosh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_cosh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_degrees t u = '[ '("_data", AttrReq (t u))]

_degrees ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_degrees t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_degrees args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_degrees t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_depth_to_space t u =
     '[ '("block_size", AttrReq Int), '("_data", AttrReq (t u))]

_depth_to_space ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_depth_to_space t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_depth_to_space args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_depth_to_space t u))
              args
        scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 Just (Anon.get #block_size fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "depth_to_space" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_diag t u =
     '[ '("k", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("_data", AttrReq (t u))]

_diag ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_diag t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_diag args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_diag t u)) args
        scalarArgs
          = catMaybes
              [("k",) . showValue <$> Anon.get #k fullArgs,
               ("axis1",) . showValue <$> Anon.get #axis1 fullArgs,
               ("axis2",) . showValue <$> Anon.get #axis2 fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_dot t u =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_dot ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_dot t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_dot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_dot t u)) args
        scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$> Anon.get #transpose_a fullArgs,
               ("transpose_b",) . showValue <$> Anon.get #transpose_b fullArgs,
               ("forward_stype",) . showValue <$>
                 Anon.get #forward_stype fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_elemwise_add t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_add ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_elemwise_add t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_elemwise_add args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_elemwise_add t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "elemwise_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_elemwise_div t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_div ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_elemwise_div t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_elemwise_div args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_elemwise_div t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "elemwise_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_elemwise_mul t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_mul ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_elemwise_mul t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_elemwise_mul args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_elemwise_mul t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "elemwise_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_elemwise_sub t u =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_sub ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_elemwise_sub t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_elemwise_sub args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_elemwise_sub t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "elemwise_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_erf t u = '[ '("_data", AttrReq (t u))]

_erf ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_erf t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_erf args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_erf t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "erf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_erfinv t u = '[ '("_data", AttrReq (t u))]

_erfinv ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_erfinv t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_erfinv args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_erfinv t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "erfinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_exp t u = '[ '("_data", AttrReq (t u))]

_exp ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_exp t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_exp args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_exp t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "exp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_expand_dims t u =
     '[ '("axis", AttrReq Int), '("_data", AttrReq (t u))]

_expand_dims ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_expand_dims t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_expand_dims args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_expand_dims t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Just (Anon.get #axis fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "expand_dims" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_expm1 t u = '[ '("_data", AttrReq (t u))]

_expm1 ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_expm1 t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_expm1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_expm1 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_fill_element_0index t u =
     '[ '("lhs", AttrOpt (t u)), '("mhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_fill_element_0index ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList_fill_element_0index t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
_fill_element_0index args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_fill_element_0index t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("mhs",) . toRaw <$> Anon.get #mhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "fill_element_0index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_fix t u = '[ '("_data", AttrReq (t u))]

_fix ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_fix t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_fix args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_fix t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "fix" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_floor t u = '[ '("_data", AttrReq (t u))]

_floor ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_floor t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_floor args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_floor t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "floor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_ftml_update t u =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Double),
        '("t", AttrReq Int), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float), '("clip_grad", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("d", AttrOpt (t u)), '("v", AttrOpt (t u)),
        '("z", AttrOpt (t u))]

_ftml_update ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_ftml_update t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_ftml_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ftml_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("t",) . showValue <$> Just (Anon.get #t fullArgs),
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_grad",) . showValue <$> Anon.get #clip_grad fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("d",) . toRaw <$> Anon.get #d fullArgs,
               ("v",) . toRaw <$> Anon.get #v fullArgs,
               ("z",) . toRaw <$> Anon.get #z fullArgs]
      in
      applyRaw "ftml_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_ftrl_update t u =
     '[ '("lr", AttrReq Float), '("lamda1", AttrOpt Float),
        '("beta", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("z", AttrOpt (t u)),
        '("n", AttrOpt (t u))]

_ftrl_update ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_ftrl_update t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_ftrl_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ftrl_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("lamda1",) . showValue <$> Anon.get #lamda1 fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("z",) . toRaw <$> Anon.get #z fullArgs,
               ("n",) . toRaw <$> Anon.get #n fullArgs]
      in
      applyRaw "ftrl_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_gamma t u = '[ '("_data", AttrReq (t u))]

_gamma ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_gamma t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_gamma args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_gamma t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_gammaln t u = '[ '("_data", AttrReq (t u))]

_gammaln ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_gammaln t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_gammaln args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_gammaln t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "gammaln" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_gather_nd t u =
     '[ '("_data", AttrReq (t u)), '("indices", AttrOpt (t u))]

_gather_nd ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_gather_nd t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_gather_nd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_gather_nd t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "gather_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_hard_sigmoid t u =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("_data", AttrReq (t u))]

_hard_sigmoid ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_hard_sigmoid t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_hard_sigmoid args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_hard_sigmoid t u))
              args
        scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> Anon.get #alpha fullArgs,
               ("beta",) . showValue <$> Anon.get #beta fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "hard_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_im2col t u =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("_data", AttrReq (t u))]

_im2col ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_im2col t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_im2col args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_im2col t u)) args
        scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> Just (Anon.get #kernel fullArgs),
               ("stride",) . showValue <$> Anon.get #stride fullArgs,
               ("dilate",) . showValue <$> Anon.get #dilate fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "im2col" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_khatri_rao t u = '[ '("args", AttrReq [t u])]

_khatri_rao ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_khatri_rao t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_khatri_rao args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_khatri_rao t u)) args
        scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #args fullArgs :: [RawTensor t]
      in applyRaw "khatri_rao" scalarArgs (Right tensorVarArgs)

type ParameterList_lamb_update_phase1 t u =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u))]

_lamb_update_phase1 ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList_lamb_update_phase1 t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
_lamb_update_phase1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_lamb_update_phase1 t u))
              args
        scalarArgs
          = catMaybes
              [("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("t",) . showValue <$> Just (Anon.get #t fullArgs),
               ("bias_correction",) . showValue <$>
                 Anon.get #bias_correction fullArgs,
               ("wd",) . showValue <$> Just (Anon.get #wd fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mean",) . toRaw <$> Anon.get #mean fullArgs,
               ("var",) . toRaw <$> Anon.get #var fullArgs]
      in
      applyRaw "lamb_update_phase1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_lamb_update_phase2 t u =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("r1", AttrOpt (t u)),
        '("r2", AttrOpt (t u))]

_lamb_update_phase2 ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList_lamb_update_phase2 t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
_lamb_update_phase2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_lamb_update_phase2 t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("lower_bound",) . showValue <$> Anon.get #lower_bound fullArgs,
               ("upper_bound",) . showValue <$> Anon.get #upper_bound fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("g",) . toRaw <$> Anon.get #g fullArgs,
               ("r1",) . toRaw <$> Anon.get #r1 fullArgs,
               ("r2",) . toRaw <$> Anon.get #r2 fullArgs]
      in
      applyRaw "lamb_update_phase2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_log t u = '[ '("_data", AttrReq (t u))]

_log ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_log t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_log args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_log t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_log10 t u = '[ '("_data", AttrReq (t u))]

_log10 ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_log10 t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_log10 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_log10 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_log1p t u = '[ '("_data", AttrReq (t u))]

_log1p ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_log1p t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_log1p args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_log1p t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_log2 t u = '[ '("_data", AttrReq (t u))]

_log2 ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_log2 t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_log2 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_log2 t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_log_softmax t u =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("_data", AttrReq (t u))]

_log_softmax ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_log_softmax t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_log_softmax args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_log_softmax t u))
              args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("temperature",) . showValue <$> Anon.get #temperature fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("use_length",) . showValue <$> Anon.get #use_length fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "log_softmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_logical_not t u = '[ '("_data", AttrReq (t u))]

_logical_not ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_logical_not t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_logical_not args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_logical_not t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "logical_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_make_loss t u = '[ '("_data", AttrReq (t u))]

_make_loss ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_make_loss t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_make_loss args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_make_loss t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "make_loss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_max t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_max ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_max t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_max args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_max t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mean t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_mean ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_mean t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_mean args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_mean t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_min t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_min ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_min t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_min args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_min t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_moments t u =
     '[ '("axes", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("_data", AttrReq (t u))]

_moments ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_moments t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_moments args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_moments t u)) args
        scalarArgs
          = catMaybes
              [("axes",) . showValue <$> Anon.get #axes fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "moments" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mp_lamb_update_phase1 t u =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_lamb_update_phase1 ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList_mp_lamb_update_phase1 t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
_mp_lamb_update_phase1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_mp_lamb_update_phase1 t u))
              args
        scalarArgs
          = catMaybes
              [("beta1",) . showValue <$> Anon.get #beta1 fullArgs,
               ("beta2",) . showValue <$> Anon.get #beta2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("t",) . showValue <$> Just (Anon.get #t fullArgs),
               ("bias_correction",) . showValue <$>
                 Anon.get #bias_correction fullArgs,
               ("wd",) . showValue <$> Just (Anon.get #wd fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mean",) . toRaw <$> Anon.get #mean fullArgs,
               ("var",) . toRaw <$> Anon.get #var fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs]
      in
      applyRaw "mp_lamb_update_phase1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mp_lamb_update_phase2 t u =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("r1", AttrOpt (t u)),
        '("r2", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_lamb_update_phase2 ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList_mp_lamb_update_phase2 t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
_mp_lamb_update_phase2 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_mp_lamb_update_phase2 t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("lower_bound",) . showValue <$> Anon.get #lower_bound fullArgs,
               ("upper_bound",) . showValue <$> Anon.get #upper_bound fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("g",) . toRaw <$> Anon.get #g fullArgs,
               ("r1",) . toRaw <$> Anon.get #r1 fullArgs,
               ("r2",) . toRaw <$> Anon.get #r2 fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs]
      in
      applyRaw "mp_lamb_update_phase2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mp_nag_mom_update t u =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mom", AttrOpt (t u)),
        '("weight32", AttrOpt (t u))]

_mp_nag_mom_update ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_mp_nag_mom_update t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_mp_nag_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_mp_nag_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mom",) . toRaw <$> Anon.get #mom fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs]
      in
      applyRaw "mp_nag_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mp_sgd_mom_update t u =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_sgd_mom_update ::
                   forall t u r .
                     (Tensor t, FieldsAcc (ParameterList_mp_sgd_mom_update t u) r,
                      HasCallStack, DType u) =>
                     Record r -> TensorApply (t u)
_mp_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_mp_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("lazy_update",) . showValue <$> Anon.get #lazy_update fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mom",) . toRaw <$> Anon.get #mom fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs]
      in
      applyRaw "mp_sgd_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_mp_sgd_update t u =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("weight32", AttrOpt (t u))]

_mp_sgd_update ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_mp_sgd_update t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_mp_sgd_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_mp_sgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("lazy_update",) . showValue <$> Anon.get #lazy_update fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("weight32",) . toRaw <$> Anon.get #weight32 fullArgs]
      in
      applyRaw "mp_sgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_multi_all_finite t u =
     '[ '("num_arrays", AttrOpt Int), '("init_output", AttrOpt Bool),
        '("_data", AttrReq [t u])]

_multi_all_finite ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList_multi_all_finite t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
_multi_all_finite args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_multi_all_finite t u))
              args
        scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$> Anon.get #num_arrays fullArgs,
               ("init_output",) . showValue <$> Anon.get #init_output fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "multi_all_finite" scalarArgs (Right tensorVarArgs)

type ParameterList_multi_lars t u =
     '[ '("eta", AttrReq Float), '("eps", AttrReq Float),
        '("rescale_grad", AttrOpt Float), '("lrs", AttrOpt (t u)),
        '("weights_sum_sq", AttrOpt (t u)),
        '("grads_sum_sq", AttrOpt (t u)), '("wds", AttrOpt (t u))]

_multi_lars ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_multi_lars t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_multi_lars args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_multi_lars t u)) args
        scalarArgs
          = catMaybes
              [("eta",) . showValue <$> Just (Anon.get #eta fullArgs),
               ("eps",) . showValue <$> Just (Anon.get #eps fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lrs",) . toRaw <$> Anon.get #lrs fullArgs,
               ("weights_sum_sq",) . toRaw <$> Anon.get #weights_sum_sq fullArgs,
               ("grads_sum_sq",) . toRaw <$> Anon.get #grads_sum_sq fullArgs,
               ("wds",) . toRaw <$> Anon.get #wds fullArgs]
      in
      applyRaw "multi_lars" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_multi_mp_sgd_mom_update t u =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_multi_mp_sgd_mom_update ::
                         forall t u r .
                           (Tensor t, FieldsAcc (ParameterList_multi_mp_sgd_mom_update t u) r,
                            HasCallStack, DType u) =>
                           Record r -> TensorApply (t u)
_multi_mp_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_multi_mp_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "multi_mp_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type ParameterList_multi_mp_sgd_update t u =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_multi_mp_sgd_update ::
                     forall t u r .
                       (Tensor t, FieldsAcc (ParameterList_multi_mp_sgd_update t u) r,
                        HasCallStack, DType u) =>
                       Record r -> TensorApply (t u)
_multi_mp_sgd_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_multi_mp_sgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "multi_mp_sgd_update" scalarArgs (Right tensorVarArgs)

type ParameterList_multi_sgd_mom_update t u =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_multi_sgd_mom_update ::
                      forall t u r .
                        (Tensor t, FieldsAcc (ParameterList_multi_sgd_mom_update t u) r,
                         HasCallStack, DType u) =>
                        Record r -> TensorApply (t u)
_multi_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_multi_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "multi_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type ParameterList_multi_sgd_update t u =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_multi_sgd_update ::
                  forall t u r .
                    (Tensor t, FieldsAcc (ParameterList_multi_sgd_update t u) r,
                     HasCallStack, DType u) =>
                    Record r -> TensorApply (t u)
_multi_sgd_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_multi_sgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> Just (Anon.get #lrs fullArgs),
               ("wds",) . showValue <$> Just (Anon.get #wds fullArgs),
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "multi_sgd_update" scalarArgs (Right tensorVarArgs)

type ParameterList_multi_sum_sq t u =
     '[ '("num_arrays", AttrReq Int), '("_data", AttrReq [t u])]

_multi_sum_sq ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_multi_sum_sq t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_multi_sum_sq args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_multi_sum_sq t u))
              args
        scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 Just (Anon.get #num_arrays fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "multi_sum_sq" scalarArgs (Right tensorVarArgs)

type ParameterList_nag_mom_update t u =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mom", AttrOpt (t u))]

_nag_mom_update ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_nag_mom_update t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_nag_mom_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_nag_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mom",) . toRaw <$> Anon.get #mom fullArgs]
      in
      applyRaw "nag_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_nanprod t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_nanprod ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_nanprod t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_nanprod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_nanprod t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "nanprod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_nansum t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_nansum ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_nansum t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_nansum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_nansum t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "nansum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_negative t u = '[ '("_data", AttrReq (t u))]

_negative ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList_negative t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
_negative args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_negative t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "negative" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_norm t u =
     '[ '("ord", AttrOpt Int), '("axis", AttrOpt (Maybe [Int])),
        '("out_dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("keepdims", AttrOpt Bool), '("_data", AttrReq (t u))]

_norm ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_norm t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_norm args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_norm t u)) args
        scalarArgs
          = catMaybes
              [("ord",) . showValue <$> Anon.get #ord fullArgs,
               ("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("out_dtype",) . showValue <$> Anon.get #out_dtype fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_one_hot t u =
     '[ '("depth", AttrReq Int), '("on_value", AttrOpt Double),
        '("off_value", AttrOpt Double),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("indices", AttrOpt (t u))]

_one_hot ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_one_hot t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_one_hot args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_one_hot t u)) args
        scalarArgs
          = catMaybes
              [("depth",) . showValue <$> Just (Anon.get #depth fullArgs),
               ("on_value",) . showValue <$> Anon.get #on_value fullArgs,
               ("off_value",) . showValue <$> Anon.get #off_value fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "one_hot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_ones_like t u = '[ '("_data", AttrReq (t u))]

_ones_like ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_ones_like t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_ones_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ones_like t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "ones_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_pick t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("mode", AttrOpt (EnumType '["clip", "wrap"])),
        '("_data", AttrReq (t u)), '("index", AttrOpt (t u))]

_pick ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_pick t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_pick args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_pick t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("mode",) . showValue <$> Anon.get #mode fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("index",) . toRaw <$> Anon.get #index fullArgs]
      in
      applyRaw "pick" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_preloaded_multi_mp_sgd_mom_update t u =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_preloaded_multi_mp_sgd_mom_update ::
                                   forall t u r .
                                     (Tensor t,
                                      FieldsAcc
                                        (ParameterList_preloaded_multi_mp_sgd_mom_update t u)
                                        r,
                                      HasCallStack, DType u) =>
                                     Record r -> TensorApply (t u)
_preloaded_multi_mp_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_preloaded_multi_mp_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "preloaded_multi_mp_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type ParameterList_preloaded_multi_mp_sgd_update t u =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_preloaded_multi_mp_sgd_update ::
                               forall t u r .
                                 (Tensor t,
                                  FieldsAcc (ParameterList_preloaded_multi_mp_sgd_update t u) r,
                                  HasCallStack, DType u) =>
                                 Record r -> TensorApply (t u)
_preloaded_multi_mp_sgd_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_preloaded_multi_mp_sgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "preloaded_multi_mp_sgd_update" scalarArgs
        (Right tensorVarArgs)

type ParameterList_preloaded_multi_sgd_mom_update t u =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_preloaded_multi_sgd_mom_update ::
                                forall t u r .
                                  (Tensor t,
                                   FieldsAcc (ParameterList_preloaded_multi_sgd_mom_update t u) r,
                                   HasCallStack, DType u) =>
                                  Record r -> TensorApply (t u)
_preloaded_multi_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_preloaded_multi_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "preloaded_multi_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type ParameterList_preloaded_multi_sgd_update t u =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("_data", AttrReq [t u])]

_preloaded_multi_sgd_update ::
                            forall t u r .
                              (Tensor t,
                               FieldsAcc (ParameterList_preloaded_multi_sgd_update t u) r,
                               HasCallStack, DType u) =>
                              Record r -> TensorApply (t u)
_preloaded_multi_sgd_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_preloaded_multi_sgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("num_weights",) . showValue <$> Anon.get #num_weights fullArgs]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in
      applyRaw "preloaded_multi_sgd_update" scalarArgs
        (Right tensorVarArgs)

type ParameterList_prod t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_prod ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_prod t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_prod args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_prod t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_radians t u = '[ '("_data", AttrReq (t u))]

_radians ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_radians t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_radians args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_radians t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_rcbrt t u = '[ '("_data", AttrReq (t u))]

_rcbrt ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_rcbrt t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_rcbrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_rcbrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "rcbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_reciprocal t u = '[ '("_data", AttrReq (t u))]

_reciprocal ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_reciprocal t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_reciprocal args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_reciprocal t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_relu t u = '[ '("_data", AttrReq (t u))]

_relu ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_relu t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_relu args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_relu t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_repeat t u =
     '[ '("repeats", AttrReq Int), '("axis", AttrOpt (Maybe Int)),
        '("_data", AttrReq (t u))]

_repeat ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_repeat t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_repeat args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_repeat t u)) args
        scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> Just (Anon.get #repeats fullArgs),
               ("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "repeat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_reset_arrays t u =
     '[ '("num_arrays", AttrReq Int), '("_data", AttrReq [t u])]

_reset_arrays ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_reset_arrays t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_reset_arrays args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_reset_arrays t u))
              args
        scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 Just (Anon.get #num_arrays fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "reset_arrays" scalarArgs (Right tensorVarArgs)

type ParameterList_reshape_like t u =
     '[ '("lhs_begin", AttrOpt (Maybe Int)),
        '("lhs_end", AttrOpt (Maybe Int)),
        '("rhs_begin", AttrOpt (Maybe Int)),
        '("rhs_end", AttrOpt (Maybe Int)), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_reshape_like ::
              forall t u r .
                (Tensor t, FieldsAcc (ParameterList_reshape_like t u) r,
                 HasCallStack, DType u) =>
                Record r -> TensorApply (t u)
_reshape_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_reshape_like t u))
              args
        scalarArgs
          = catMaybes
              [("lhs_begin",) . showValue <$> Anon.get #lhs_begin fullArgs,
               ("lhs_end",) . showValue <$> Anon.get #lhs_end fullArgs,
               ("rhs_begin",) . showValue <$> Anon.get #rhs_begin fullArgs,
               ("rhs_end",) . showValue <$> Anon.get #rhs_end fullArgs]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> Anon.get #lhs fullArgs,
               ("rhs",) . toRaw <$> Anon.get #rhs fullArgs]
      in
      applyRaw "reshape_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_reverse t u =
     '[ '("axis", AttrReq [Int]), '("_data", AttrReq (t u))]

_reverse ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_reverse t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_reverse args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_reverse t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Just (Anon.get #axis fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "reverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_rint t u = '[ '("_data", AttrReq (t u))]

_rint ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_rint t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_rint args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_rint t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "rint" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_rmsprop_update t u =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("n", AttrOpt (t u))]

_rmsprop_update ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_rmsprop_update t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_rmsprop_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_rmsprop_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("gamma1",) . showValue <$> Anon.get #gamma1 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("clip_weights",) . showValue <$> Anon.get #clip_weights fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("n",) . toRaw <$> Anon.get #n fullArgs]
      in
      applyRaw "rmsprop_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_rmspropalex_update t u =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("gamma2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("n", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("delta", AttrOpt (t u))]

_rmspropalex_update ::
                    forall t u r .
                      (Tensor t, FieldsAcc (ParameterList_rmspropalex_update t u) r,
                       HasCallStack, DType u) =>
                      Record r -> TensorApply (t u)
_rmspropalex_update args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_rmspropalex_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("gamma1",) . showValue <$> Anon.get #gamma1 fullArgs,
               ("gamma2",) . showValue <$> Anon.get #gamma2 fullArgs,
               ("epsilon",) . showValue <$> Anon.get #epsilon fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("clip_weights",) . showValue <$> Anon.get #clip_weights fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("n",) . toRaw <$> Anon.get #n fullArgs,
               ("g",) . toRaw <$> Anon.get #g fullArgs,
               ("delta",) . toRaw <$> Anon.get #delta fullArgs]
      in
      applyRaw "rmspropalex_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_round t u = '[ '("_data", AttrReq (t u))]

_round ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_round t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_round args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_round t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "round" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_rsqrt t u = '[ '("_data", AttrReq (t u))]

_rsqrt ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_rsqrt t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_rsqrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_rsqrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "rsqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_scatter_nd t u =
     '[ '("shape", AttrReq [Int]), '("_data", AttrReq (t u)),
        '("indices", AttrOpt (t u))]

_scatter_nd ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_scatter_nd t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_scatter_nd args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_scatter_nd t u)) args
        scalarArgs
          = catMaybes
              [("shape",) . showValue <$> Just (Anon.get #shape fullArgs)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "scatter_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sgd_mom_update t u =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u))]

_sgd_mom_update ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_sgd_mom_update t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_sgd_mom_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sgd_mom_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("lazy_update",) . showValue <$> Anon.get #lazy_update fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mom",) . toRaw <$> Anon.get #mom fullArgs]
      in
      applyRaw "sgd_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sgd_update t u =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u))]

_sgd_update ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_sgd_update t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_sgd_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sgd_update t u)) args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("lazy_update",) . showValue <$> Anon.get #lazy_update fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs]
      in
      applyRaw "sgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_shape_array t u = '[ '("_data", AttrReq (t u))]

_shape_array ::
             forall t u r .
               (Tensor t, FieldsAcc (ParameterList_shape_array t u) r,
                HasCallStack, DType u) =>
               Record r -> TensorApply (t u)
_shape_array args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_shape_array t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "shape_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sigmoid t u = '[ '("_data", AttrReq (t u))]

_sigmoid ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_sigmoid t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_sigmoid args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sigmoid t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sign t u = '[ '("_data", AttrReq (t u))]

_sign ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_sign t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_sign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sign t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_signsgd_update t u =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u))]

_signsgd_update ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_signsgd_update t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_signsgd_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_signsgd_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs]
      in
      applyRaw "signsgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_signum_update t u =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("wd_lh", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u))]

_signum_update ::
               forall t u r .
                 (Tensor t, FieldsAcc (ParameterList_signum_update t u) r,
                  HasCallStack, DType u) =>
                 Record r -> TensorApply (t u)
_signum_update args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_signum_update t u))
              args
        scalarArgs
          = catMaybes
              [("lr",) . showValue <$> Just (Anon.get #lr fullArgs),
               ("momentum",) . showValue <$> Anon.get #momentum fullArgs,
               ("wd",) . showValue <$> Anon.get #wd fullArgs,
               ("rescale_grad",) . showValue <$> Anon.get #rescale_grad fullArgs,
               ("clip_gradient",) . showValue <$>
                 Anon.get #clip_gradient fullArgs,
               ("wd_lh",) . showValue <$> Anon.get #wd_lh fullArgs]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> Anon.get #weight fullArgs,
               ("grad",) . toRaw <$> Anon.get #grad fullArgs,
               ("mom",) . toRaw <$> Anon.get #mom fullArgs]
      in
      applyRaw "signum_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sin t u = '[ '("_data", AttrReq (t u))]

_sin ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_sin t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_sin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sin t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sinh t u = '[ '("_data", AttrReq (t u))]

_sinh ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_sinh t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_sinh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sinh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_size_array t u = '[ '("_data", AttrReq (t u))]

_size_array ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_size_array t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_size_array args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_size_array t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "size_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_slice t u =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("_data", AttrReq (t u))]

_slice ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_slice t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_slice args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_slice t u)) args
        scalarArgs
          = catMaybes
              [("begin",) . showValue <$> Just (Anon.get #begin fullArgs),
               ("end",) . showValue <$> Just (Anon.get #end fullArgs),
               ("step",) . showValue <$> Anon.get #step fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_slice_axis t u =
     '[ '("axis", AttrReq Int), '("begin", AttrReq Int),
        '("end", AttrReq (Maybe Int)), '("_data", AttrReq (t u))]

_slice_axis ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_slice_axis t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_slice_axis args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_slice_axis t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Just (Anon.get #axis fullArgs),
               ("begin",) . showValue <$> Just (Anon.get #begin fullArgs),
               ("end",) . showValue <$> Just (Anon.get #end fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "slice_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_slice_like t u =
     '[ '("axes", AttrOpt [Int]), '("_data", AttrReq (t u)),
        '("shape_like", AttrOpt (t u))]

_slice_like ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_slice_like t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_slice_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_slice_like t u)) args
        scalarArgs
          = catMaybes [("axes",) . showValue <$> Anon.get #axes fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("shape_like",) . toRaw <$> Anon.get #shape_like fullArgs]
      in
      applyRaw "slice_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_smooth_l1 t u =
     '[ '("scalar", AttrOpt Float), '("_data", AttrReq (t u))]

_smooth_l1 ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_smooth_l1 t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_smooth_l1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_smooth_l1 t u)) args
        scalarArgs
          = catMaybes [("scalar",) . showValue <$> Anon.get #scalar fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "smooth_l1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_softmax t u =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("_data", AttrReq (t u)),
        '("length", AttrOpt (t u))]

_softmax ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_softmax t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_softmax args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_softmax t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("temperature",) . showValue <$> Anon.get #temperature fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("use_length",) . showValue <$> Anon.get #use_length fullArgs]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("length",) . toRaw <$> Anon.get #length fullArgs]
      in
      applyRaw "softmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_softmax_cross_entropy t u =
     '[ '("_data", AttrReq (t u)), '("label", AttrOpt (t u))]

_softmax_cross_entropy ::
                       forall t u r .
                         (Tensor t, FieldsAcc (ParameterList_softmax_cross_entropy t u) r,
                          HasCallStack, DType u) =>
                         Record r -> TensorApply (t u)
_softmax_cross_entropy args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_softmax_cross_entropy t u))
              args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> Just (Anon.get #_data fullArgs),
               ("label",) . toRaw <$> Anon.get #label fullArgs]
      in
      applyRaw "softmax_cross_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_softmin t u =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("_data", AttrReq (t u))]

_softmin ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_softmin t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_softmin args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_softmin t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("temperature",) . showValue <$> Anon.get #temperature fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("use_length",) . showValue <$> Anon.get #use_length fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "softmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_softsign t u = '[ '("_data", AttrReq (t u))]

_softsign ::
          forall t u r .
            (Tensor t, FieldsAcc (ParameterList_softsign t u) r, HasCallStack,
             DType u) =>
            Record r -> TensorApply (t u)
_softsign args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_softsign t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "softsign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sort t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("_data", AttrReq (t u))]

_sort ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_sort t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_sort args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sort t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("is_ascend",) . showValue <$> Anon.get #is_ascend fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sort" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_space_to_depth t u =
     '[ '("block_size", AttrReq Int), '("_data", AttrReq (t u))]

_space_to_depth ::
                forall t u r .
                  (Tensor t, FieldsAcc (ParameterList_space_to_depth t u) r,
                   HasCallStack, DType u) =>
                  Record r -> TensorApply (t u)
_space_to_depth args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_space_to_depth t u))
              args
        scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 Just (Anon.get #block_size fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "space_to_depth" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_sqrt t u = '[ '("_data", AttrReq (t u))]

_sqrt ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_sqrt t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_sqrt args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sqrt t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_square t u = '[ '("_data", AttrReq (t u))]

_square ::
        forall t u r .
          (Tensor t, FieldsAcc (ParameterList_square t u) r, HasCallStack,
           DType u) =>
          Record r -> TensorApply (t u)
_square args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_square t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_squeeze t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("_data", AttrReq (t u))]

_squeeze ::
         forall t u r .
           (Tensor t, FieldsAcc (ParameterList_squeeze t u) r, HasCallStack,
            DType u) =>
           Record r -> TensorApply (t u)
_squeeze args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_squeeze t u)) args
        scalarArgs
          = catMaybes [("axis",) . showValue <$> Anon.get #axis fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_stack t u =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("_data", AttrReq [t u])]

_stack ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_stack t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_stack args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_stack t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("num_args",) . showValue <$> Just (Anon.get #num_args fullArgs)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = map toRaw $ Anon.get #_data fullArgs :: [RawTensor t]
      in applyRaw "stack" scalarArgs (Right tensorVarArgs)

type ParameterList_sum t u =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("_data", AttrReq (t u))]

_sum ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_sum t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_sum args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_sum t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("keepdims",) . showValue <$> Anon.get #keepdims fullArgs,
               ("exclude",) . showValue <$> Anon.get #exclude fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_take t u =
     '[ '("axis", AttrOpt Int),
        '("mode", AttrOpt (EnumType '["clip", "raise", "wrap"])),
        '("a", AttrOpt (t u)), '("indices", AttrOpt (t u))]

_take ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_take t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_take args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_take t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("mode",) . showValue <$> Anon.get #mode fullArgs]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> Anon.get #a fullArgs,
               ("indices",) . toRaw <$> Anon.get #indices fullArgs]
      in
      applyRaw "take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_tan t u = '[ '("_data", AttrReq (t u))]

_tan ::
     forall t u r .
       (Tensor t, FieldsAcc (ParameterList_tan t u) r, HasCallStack,
        DType u) =>
       Record r -> TensorApply (t u)
_tan args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_tan t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_tanh t u = '[ '("_data", AttrReq (t u))]

_tanh ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_tanh t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_tanh args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_tanh t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_tile t u =
     '[ '("reps", AttrReq [Int]), '("_data", AttrReq (t u))]

_tile ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_tile t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_tile args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_tile t u)) args
        scalarArgs
          = catMaybes
              [("reps",) . showValue <$> Just (Anon.get #reps fullArgs)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "tile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_topk t u =
     '[ '("axis", AttrOpt (Maybe Int)), '("k", AttrOpt Int),
        '("ret_typ",
          AttrOpt (EnumType '["both", "indices", "mask", "value"])),
        '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("_data", AttrReq (t u))]

_topk ::
      forall t u r .
        (Tensor t, FieldsAcc (ParameterList_topk t u) r, HasCallStack,
         DType u) =>
        Record r -> TensorApply (t u)
_topk args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_topk t u)) args
        scalarArgs
          = catMaybes
              [("axis",) . showValue <$> Anon.get #axis fullArgs,
               ("k",) . showValue <$> Anon.get #k fullArgs,
               ("ret_typ",) . showValue <$> Anon.get #ret_typ fullArgs,
               ("is_ascend",) . showValue <$> Anon.get #is_ascend fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "topk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_transpose t u =
     '[ '("axes", AttrOpt [Int]), '("_data", AttrReq (t u))]

_transpose ::
           forall t u r .
             (Tensor t, FieldsAcc (ParameterList_transpose t u) r, HasCallStack,
              DType u) =>
             Record r -> TensorApply (t u)
_transpose args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_transpose t u)) args
        scalarArgs
          = catMaybes [("axes",) . showValue <$> Anon.get #axes fullArgs]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "transpose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_trunc t u = '[ '("_data", AttrReq (t u))]

_trunc ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_trunc t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_trunc args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_trunc t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "trunc" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_where t u =
     '[ '("condition", AttrOpt (t u)), '("x", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

_where ::
       forall t u r .
         (Tensor t, FieldsAcc (ParameterList_where t u) r, HasCallStack,
          DType u) =>
         Record r -> TensorApply (t u)
_where args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_where t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> Anon.get #condition fullArgs,
               ("x",) . toRaw <$> Anon.get #x fullArgs,
               ("y",) . toRaw <$> Anon.get #y fullArgs]
      in
      applyRaw "where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type ParameterList_zeros_like t u = '[ '("_data", AttrReq (t u))]

_zeros_like ::
            forall t u r .
              (Tensor t, FieldsAcc (ParameterList_zeros_like t u) r,
               HasCallStack, DType u) =>
              Record r -> TensorApply (t u)
_zeros_like args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_zeros_like t u)) args
        scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> Just (Anon.get #_data fullArgs)]
      in
      applyRaw "zeros_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))