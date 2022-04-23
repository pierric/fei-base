{-# LANGUAGE AllowAmbiguousTypes, PolyKinds, TypeOperators,
  TypeApplications #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.Base.DataIter where
import RIO
import RIO.List
import RIO.List.Partial ((!!))
import MXNet.Base.Raw
import MXNet.Base.Core.Spec
import MXNet.Base.Core.Enum
import MXNet.Base.Tensor.Class
import MXNet.Base.Types (DType)
import Data.Maybe (catMaybes, fromMaybe)
import Data.Record.Anon.Simple (Record)
import qualified Data.Record.Anon.Simple as Anon

type ParameterList_CSVIter =
     '[ '("data_csv", AttrReq Text), '("data_shape", AttrReq [Int]),
        '("label_csv", AttrOpt Text), '("label_shape", AttrOpt [Int]),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"])))]

_CSVIter ::
         forall r . (FieldsAcc ParameterList_CSVIter r, HasCallStack) =>
           Record r -> IO DataIterHandle
_CSVIter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_CSVIter)) args
        scalarArgs
          = catMaybes
              [("data_csv",) . showValue <$> Just (Anon.get #data_csv fullArgs),
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("label_csv",) . showValue <$> Anon.get #label_csv fullArgs,
               ("label_shape",) . showValue <$> Anon.get #label_shape fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 0)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageDetRecordIter =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("aug_seq", AttrOpt Text), '("label_width", AttrOpt Int),
        '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("label_pad_width", AttrOpt Int),
        '("label_pad_value", AttrOpt Float), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop_prob", AttrOpt Float),
        '("min_crop_scales", AttrOpt [Float]),
        '("max_crop_scales", AttrOpt [Float]),
        '("min_crop_aspect_ratios", AttrOpt [Float]),
        '("max_crop_aspect_ratios", AttrOpt [Float]),
        '("min_crop_overlaps", AttrOpt [Float]),
        '("max_crop_overlaps", AttrOpt [Float]),
        '("min_crop_sample_coverages", AttrOpt [Float]),
        '("max_crop_sample_coverages", AttrOpt [Float]),
        '("min_crop_object_coverages", AttrOpt [Float]),
        '("max_crop_object_coverages", AttrOpt [Float]),
        '("num_crop_sampler", AttrOpt Int),
        '("crop_emit_mode", AttrOpt (EnumType '["center", "overlap"])),
        '("emit_overlap_thresh", AttrOpt Float),
        '("max_crop_trials", AttrOpt [Int]),
        '("rand_pad_prob", AttrOpt Float),
        '("max_pad_scale", AttrOpt Float),
        '("max_random_hue", AttrOpt Int),
        '("random_hue_prob", AttrOpt Float),
        '("max_random_saturation", AttrOpt Int),
        '("random_saturation_prob", AttrOpt Float),
        '("max_random_illumination", AttrOpt Int),
        '("random_illumination_prob", AttrOpt Float),
        '("max_random_contrast", AttrOpt Float),
        '("random_contrast_prob", AttrOpt Float),
        '("rand_mirror_prob", AttrOpt Float), '("fill_value", AttrOpt Int),
        '("inter_method", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("resize_mode", AttrOpt (EnumType '["fit", "force", "shrink"])),
        '("seed", AttrOpt Int), '("mean_img", AttrOpt Text),
        '("mean_r", AttrOpt Float), '("mean_g", AttrOpt Float),
        '("mean_b", AttrOpt Float), '("mean_a", AttrOpt Float),
        '("std_r", AttrOpt Float), '("std_g", AttrOpt Float),
        '("std_b", AttrOpt Float), '("std_a", AttrOpt Float),
        '("scale", AttrOpt Float), '("verbose", AttrOpt Bool)]

_ImageDetRecordIter ::
                    forall r .
                      (FieldsAcc ParameterList_ImageDetRecordIter r, HasCallStack) =>
                      Record r -> IO DataIterHandle
_ImageDetRecordIter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ImageDetRecordIter))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("label_pad_width",) . showValue <$>
                 Anon.get #label_pad_width fullArgs,
               ("label_pad_value",) . showValue <$>
                 Anon.get #label_pad_value fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop_prob",) . showValue <$>
                 Anon.get #rand_crop_prob fullArgs,
               ("min_crop_scales",) . showValue <$>
                 Anon.get #min_crop_scales fullArgs,
               ("max_crop_scales",) . showValue <$>
                 Anon.get #max_crop_scales fullArgs,
               ("min_crop_aspect_ratios",) . showValue <$>
                 Anon.get #min_crop_aspect_ratios fullArgs,
               ("max_crop_aspect_ratios",) . showValue <$>
                 Anon.get #max_crop_aspect_ratios fullArgs,
               ("min_crop_overlaps",) . showValue <$>
                 Anon.get #min_crop_overlaps fullArgs,
               ("max_crop_overlaps",) . showValue <$>
                 Anon.get #max_crop_overlaps fullArgs,
               ("min_crop_sample_coverages",) . showValue <$>
                 Anon.get #min_crop_sample_coverages fullArgs,
               ("max_crop_sample_coverages",) . showValue <$>
                 Anon.get #max_crop_sample_coverages fullArgs,
               ("min_crop_object_coverages",) . showValue <$>
                 Anon.get #min_crop_object_coverages fullArgs,
               ("max_crop_object_coverages",) . showValue <$>
                 Anon.get #max_crop_object_coverages fullArgs,
               ("num_crop_sampler",) . showValue <$>
                 Anon.get #num_crop_sampler fullArgs,
               ("crop_emit_mode",) . showValue <$>
                 Anon.get #crop_emit_mode fullArgs,
               ("emit_overlap_thresh",) . showValue <$>
                 Anon.get #emit_overlap_thresh fullArgs,
               ("max_crop_trials",) . showValue <$>
                 Anon.get #max_crop_trials fullArgs,
               ("rand_pad_prob",) . showValue <$>
                 Anon.get #rand_pad_prob fullArgs,
               ("max_pad_scale",) . showValue <$>
                 Anon.get #max_pad_scale fullArgs,
               ("max_random_hue",) . showValue <$>
                 Anon.get #max_random_hue fullArgs,
               ("random_hue_prob",) . showValue <$>
                 Anon.get #random_hue_prob fullArgs,
               ("max_random_saturation",) . showValue <$>
                 Anon.get #max_random_saturation fullArgs,
               ("random_saturation_prob",) . showValue <$>
                 Anon.get #random_saturation_prob fullArgs,
               ("max_random_illumination",) . showValue <$>
                 Anon.get #max_random_illumination fullArgs,
               ("random_illumination_prob",) . showValue <$>
                 Anon.get #random_illumination_prob fullArgs,
               ("max_random_contrast",) . showValue <$>
                 Anon.get #max_random_contrast fullArgs,
               ("random_contrast_prob",) . showValue <$>
                 Anon.get #random_contrast_prob fullArgs,
               ("rand_mirror_prob",) . showValue <$>
                 Anon.get #rand_mirror_prob fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("resize_mode",) . showValue <$> Anon.get #resize_mode fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("mean_img",) . showValue <$> Anon.get #mean_img fullArgs,
               ("mean_r",) . showValue <$> Anon.get #mean_r fullArgs,
               ("mean_g",) . showValue <$> Anon.get #mean_g fullArgs,
               ("mean_b",) . showValue <$> Anon.get #mean_b fullArgs,
               ("mean_a",) . showValue <$> Anon.get #mean_a fullArgs,
               ("std_r",) . showValue <$> Anon.get #std_r fullArgs,
               ("std_g",) . showValue <$> Anon.get #std_g fullArgs,
               ("std_b",) . showValue <$> Anon.get #std_b fullArgs,
               ("std_a",) . showValue <$> Anon.get #std_a fullArgs,
               ("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 1)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageRecordIter_v1 =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("path_imgidx", AttrOpt Text), '("aug_seq", AttrOpt Text),
        '("label_width", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("device_id", AttrOpt Int), '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("seed_aug", AttrOpt (Maybe Int)), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop", AttrOpt Bool),
        '("random_resized_crop", AttrOpt Bool),
        '("max_rotate_angle", AttrOpt Int),
        '("max_aspect_ratio", AttrOpt Float),
        '("min_aspect_ratio", AttrOpt (Maybe Float)),
        '("max_shear_ratio", AttrOpt Float),
        '("max_crop_size", AttrOpt Int), '("min_crop_size", AttrOpt Int),
        '("max_random_scale", AttrOpt Float),
        '("min_random_scale", AttrOpt Float),
        '("max_random_area", AttrOpt Float),
        '("min_random_area", AttrOpt Float),
        '("max_img_size", AttrOpt Float), '("min_img_size", AttrOpt Float),
        '("brightness", AttrOpt Float), '("contrast", AttrOpt Float),
        '("saturation", AttrOpt Float), '("pca_noise", AttrOpt Float),
        '("random_h", AttrOpt Int), '("random_s", AttrOpt Int),
        '("random_l", AttrOpt Int), '("rotate", AttrOpt Int),
        '("fill_value", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("inter_method", AttrOpt Int), '("pad", AttrOpt Int),
        '("seed", AttrOpt Int), '("mirror", AttrOpt Bool),
        '("rand_mirror", AttrOpt Bool), '("mean_img", AttrOpt Text),
        '("mean_r", AttrOpt Float), '("mean_g", AttrOpt Float),
        '("mean_b", AttrOpt Float), '("mean_a", AttrOpt Float),
        '("std_r", AttrOpt Float), '("std_g", AttrOpt Float),
        '("std_b", AttrOpt Float), '("std_a", AttrOpt Float),
        '("scale", AttrOpt Float), '("max_random_contrast", AttrOpt Float),
        '("max_random_illumination", AttrOpt Float),
        '("verbose", AttrOpt Bool)]

_ImageRecordIter_v1 ::
                    forall r .
                      (FieldsAcc ParameterList_ImageRecordIter_v1 r, HasCallStack) =>
                      Record r -> IO DataIterHandle
_ImageRecordIter_v1 args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ImageRecordIter_v1))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("path_imgidx",) . showValue <$> Anon.get #path_imgidx fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("device_id",) . showValue <$> Anon.get #device_id fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("seed_aug",) . showValue <$> Anon.get #seed_aug fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop",) . showValue <$> Anon.get #rand_crop fullArgs,
               ("random_resized_crop",) . showValue <$>
                 Anon.get #random_resized_crop fullArgs,
               ("max_rotate_angle",) . showValue <$>
                 Anon.get #max_rotate_angle fullArgs,
               ("max_aspect_ratio",) . showValue <$>
                 Anon.get #max_aspect_ratio fullArgs,
               ("min_aspect_ratio",) . showValue <$>
                 Anon.get #min_aspect_ratio fullArgs,
               ("max_shear_ratio",) . showValue <$>
                 Anon.get #max_shear_ratio fullArgs,
               ("max_crop_size",) . showValue <$>
                 Anon.get #max_crop_size fullArgs,
               ("min_crop_size",) . showValue <$>
                 Anon.get #min_crop_size fullArgs,
               ("max_random_scale",) . showValue <$>
                 Anon.get #max_random_scale fullArgs,
               ("min_random_scale",) . showValue <$>
                 Anon.get #min_random_scale fullArgs,
               ("max_random_area",) . showValue <$>
                 Anon.get #max_random_area fullArgs,
               ("min_random_area",) . showValue <$>
                 Anon.get #min_random_area fullArgs,
               ("max_img_size",) . showValue <$> Anon.get #max_img_size fullArgs,
               ("min_img_size",) . showValue <$> Anon.get #min_img_size fullArgs,
               ("brightness",) . showValue <$> Anon.get #brightness fullArgs,
               ("contrast",) . showValue <$> Anon.get #contrast fullArgs,
               ("saturation",) . showValue <$> Anon.get #saturation fullArgs,
               ("pca_noise",) . showValue <$> Anon.get #pca_noise fullArgs,
               ("random_h",) . showValue <$> Anon.get #random_h fullArgs,
               ("random_s",) . showValue <$> Anon.get #random_s fullArgs,
               ("random_l",) . showValue <$> Anon.get #random_l fullArgs,
               ("rotate",) . showValue <$> Anon.get #rotate fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("mirror",) . showValue <$> Anon.get #mirror fullArgs,
               ("rand_mirror",) . showValue <$> Anon.get #rand_mirror fullArgs,
               ("mean_img",) . showValue <$> Anon.get #mean_img fullArgs,
               ("mean_r",) . showValue <$> Anon.get #mean_r fullArgs,
               ("mean_g",) . showValue <$> Anon.get #mean_g fullArgs,
               ("mean_b",) . showValue <$> Anon.get #mean_b fullArgs,
               ("mean_a",) . showValue <$> Anon.get #mean_a fullArgs,
               ("std_r",) . showValue <$> Anon.get #std_r fullArgs,
               ("std_g",) . showValue <$> Anon.get #std_g fullArgs,
               ("std_b",) . showValue <$> Anon.get #std_b fullArgs,
               ("std_a",) . showValue <$> Anon.get #std_a fullArgs,
               ("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("max_random_contrast",) . showValue <$>
                 Anon.get #max_random_contrast fullArgs,
               ("max_random_illumination",) . showValue <$>
                 Anon.get #max_random_illumination fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 2)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageRecordUInt8Iter_v1 =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("path_imgidx", AttrOpt Text), '("aug_seq", AttrOpt Text),
        '("label_width", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("device_id", AttrOpt Int), '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("seed_aug", AttrOpt (Maybe Int)), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop", AttrOpt Bool),
        '("random_resized_crop", AttrOpt Bool),
        '("max_rotate_angle", AttrOpt Int),
        '("max_aspect_ratio", AttrOpt Float),
        '("min_aspect_ratio", AttrOpt (Maybe Float)),
        '("max_shear_ratio", AttrOpt Float),
        '("max_crop_size", AttrOpt Int), '("min_crop_size", AttrOpt Int),
        '("max_random_scale", AttrOpt Float),
        '("min_random_scale", AttrOpt Float),
        '("max_random_area", AttrOpt Float),
        '("min_random_area", AttrOpt Float),
        '("max_img_size", AttrOpt Float), '("min_img_size", AttrOpt Float),
        '("brightness", AttrOpt Float), '("contrast", AttrOpt Float),
        '("saturation", AttrOpt Float), '("pca_noise", AttrOpt Float),
        '("random_h", AttrOpt Int), '("random_s", AttrOpt Int),
        '("random_l", AttrOpt Int), '("rotate", AttrOpt Int),
        '("fill_value", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("inter_method", AttrOpt Int), '("pad", AttrOpt Int)]

_ImageRecordUInt8Iter_v1 ::
                         forall r .
                           (FieldsAcc ParameterList_ImageRecordUInt8Iter_v1 r,
                            HasCallStack) =>
                           Record r -> IO DataIterHandle
_ImageRecordUInt8Iter_v1 args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_ImageRecordUInt8Iter_v1))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("path_imgidx",) . showValue <$> Anon.get #path_imgidx fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("device_id",) . showValue <$> Anon.get #device_id fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("seed_aug",) . showValue <$> Anon.get #seed_aug fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop",) . showValue <$> Anon.get #rand_crop fullArgs,
               ("random_resized_crop",) . showValue <$>
                 Anon.get #random_resized_crop fullArgs,
               ("max_rotate_angle",) . showValue <$>
                 Anon.get #max_rotate_angle fullArgs,
               ("max_aspect_ratio",) . showValue <$>
                 Anon.get #max_aspect_ratio fullArgs,
               ("min_aspect_ratio",) . showValue <$>
                 Anon.get #min_aspect_ratio fullArgs,
               ("max_shear_ratio",) . showValue <$>
                 Anon.get #max_shear_ratio fullArgs,
               ("max_crop_size",) . showValue <$>
                 Anon.get #max_crop_size fullArgs,
               ("min_crop_size",) . showValue <$>
                 Anon.get #min_crop_size fullArgs,
               ("max_random_scale",) . showValue <$>
                 Anon.get #max_random_scale fullArgs,
               ("min_random_scale",) . showValue <$>
                 Anon.get #min_random_scale fullArgs,
               ("max_random_area",) . showValue <$>
                 Anon.get #max_random_area fullArgs,
               ("min_random_area",) . showValue <$>
                 Anon.get #min_random_area fullArgs,
               ("max_img_size",) . showValue <$> Anon.get #max_img_size fullArgs,
               ("min_img_size",) . showValue <$> Anon.get #min_img_size fullArgs,
               ("brightness",) . showValue <$> Anon.get #brightness fullArgs,
               ("contrast",) . showValue <$> Anon.get #contrast fullArgs,
               ("saturation",) . showValue <$> Anon.get #saturation fullArgs,
               ("pca_noise",) . showValue <$> Anon.get #pca_noise fullArgs,
               ("random_h",) . showValue <$> Anon.get #random_h fullArgs,
               ("random_s",) . showValue <$> Anon.get #random_s fullArgs,
               ("random_l",) . showValue <$> Anon.get #random_l fullArgs,
               ("rotate",) . showValue <$> Anon.get #rotate fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 3)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageRecordIter =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("path_imgidx", AttrOpt Text), '("aug_seq", AttrOpt Text),
        '("label_width", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("device_id", AttrOpt Int), '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("seed_aug", AttrOpt (Maybe Int)), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop", AttrOpt Bool),
        '("random_resized_crop", AttrOpt Bool),
        '("max_rotate_angle", AttrOpt Int),
        '("max_aspect_ratio", AttrOpt Float),
        '("min_aspect_ratio", AttrOpt (Maybe Float)),
        '("max_shear_ratio", AttrOpt Float),
        '("max_crop_size", AttrOpt Int), '("min_crop_size", AttrOpt Int),
        '("max_random_scale", AttrOpt Float),
        '("min_random_scale", AttrOpt Float),
        '("max_random_area", AttrOpt Float),
        '("min_random_area", AttrOpt Float),
        '("max_img_size", AttrOpt Float), '("min_img_size", AttrOpt Float),
        '("brightness", AttrOpt Float), '("contrast", AttrOpt Float),
        '("saturation", AttrOpt Float), '("pca_noise", AttrOpt Float),
        '("random_h", AttrOpt Int), '("random_s", AttrOpt Int),
        '("random_l", AttrOpt Int), '("rotate", AttrOpt Int),
        '("fill_value", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("inter_method", AttrOpt Int), '("pad", AttrOpt Int),
        '("seed", AttrOpt Int), '("mirror", AttrOpt Bool),
        '("rand_mirror", AttrOpt Bool), '("mean_img", AttrOpt Text),
        '("mean_r", AttrOpt Float), '("mean_g", AttrOpt Float),
        '("mean_b", AttrOpt Float), '("mean_a", AttrOpt Float),
        '("std_r", AttrOpt Float), '("std_g", AttrOpt Float),
        '("std_b", AttrOpt Float), '("std_a", AttrOpt Float),
        '("scale", AttrOpt Float), '("max_random_contrast", AttrOpt Float),
        '("max_random_illumination", AttrOpt Float),
        '("verbose", AttrOpt Bool)]

_ImageRecordIter ::
                 forall r .
                   (FieldsAcc ParameterList_ImageRecordIter r, HasCallStack) =>
                   Record r -> IO DataIterHandle
_ImageRecordIter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ImageRecordIter))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("path_imgidx",) . showValue <$> Anon.get #path_imgidx fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("device_id",) . showValue <$> Anon.get #device_id fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("seed_aug",) . showValue <$> Anon.get #seed_aug fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop",) . showValue <$> Anon.get #rand_crop fullArgs,
               ("random_resized_crop",) . showValue <$>
                 Anon.get #random_resized_crop fullArgs,
               ("max_rotate_angle",) . showValue <$>
                 Anon.get #max_rotate_angle fullArgs,
               ("max_aspect_ratio",) . showValue <$>
                 Anon.get #max_aspect_ratio fullArgs,
               ("min_aspect_ratio",) . showValue <$>
                 Anon.get #min_aspect_ratio fullArgs,
               ("max_shear_ratio",) . showValue <$>
                 Anon.get #max_shear_ratio fullArgs,
               ("max_crop_size",) . showValue <$>
                 Anon.get #max_crop_size fullArgs,
               ("min_crop_size",) . showValue <$>
                 Anon.get #min_crop_size fullArgs,
               ("max_random_scale",) . showValue <$>
                 Anon.get #max_random_scale fullArgs,
               ("min_random_scale",) . showValue <$>
                 Anon.get #min_random_scale fullArgs,
               ("max_random_area",) . showValue <$>
                 Anon.get #max_random_area fullArgs,
               ("min_random_area",) . showValue <$>
                 Anon.get #min_random_area fullArgs,
               ("max_img_size",) . showValue <$> Anon.get #max_img_size fullArgs,
               ("min_img_size",) . showValue <$> Anon.get #min_img_size fullArgs,
               ("brightness",) . showValue <$> Anon.get #brightness fullArgs,
               ("contrast",) . showValue <$> Anon.get #contrast fullArgs,
               ("saturation",) . showValue <$> Anon.get #saturation fullArgs,
               ("pca_noise",) . showValue <$> Anon.get #pca_noise fullArgs,
               ("random_h",) . showValue <$> Anon.get #random_h fullArgs,
               ("random_s",) . showValue <$> Anon.get #random_s fullArgs,
               ("random_l",) . showValue <$> Anon.get #random_l fullArgs,
               ("rotate",) . showValue <$> Anon.get #rotate fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("mirror",) . showValue <$> Anon.get #mirror fullArgs,
               ("rand_mirror",) . showValue <$> Anon.get #rand_mirror fullArgs,
               ("mean_img",) . showValue <$> Anon.get #mean_img fullArgs,
               ("mean_r",) . showValue <$> Anon.get #mean_r fullArgs,
               ("mean_g",) . showValue <$> Anon.get #mean_g fullArgs,
               ("mean_b",) . showValue <$> Anon.get #mean_b fullArgs,
               ("mean_a",) . showValue <$> Anon.get #mean_a fullArgs,
               ("std_r",) . showValue <$> Anon.get #std_r fullArgs,
               ("std_g",) . showValue <$> Anon.get #std_g fullArgs,
               ("std_b",) . showValue <$> Anon.get #std_b fullArgs,
               ("std_a",) . showValue <$> Anon.get #std_a fullArgs,
               ("scale",) . showValue <$> Anon.get #scale fullArgs,
               ("max_random_contrast",) . showValue <$>
                 Anon.get #max_random_contrast fullArgs,
               ("max_random_illumination",) . showValue <$>
                 Anon.get #max_random_illumination fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 4)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageRecordUInt8Iter =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("path_imgidx", AttrOpt Text), '("aug_seq", AttrOpt Text),
        '("label_width", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("device_id", AttrOpt Int), '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("seed_aug", AttrOpt (Maybe Int)), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop", AttrOpt Bool),
        '("random_resized_crop", AttrOpt Bool),
        '("max_rotate_angle", AttrOpt Int),
        '("max_aspect_ratio", AttrOpt Float),
        '("min_aspect_ratio", AttrOpt (Maybe Float)),
        '("max_shear_ratio", AttrOpt Float),
        '("max_crop_size", AttrOpt Int), '("min_crop_size", AttrOpt Int),
        '("max_random_scale", AttrOpt Float),
        '("min_random_scale", AttrOpt Float),
        '("max_random_area", AttrOpt Float),
        '("min_random_area", AttrOpt Float),
        '("max_img_size", AttrOpt Float), '("min_img_size", AttrOpt Float),
        '("brightness", AttrOpt Float), '("contrast", AttrOpt Float),
        '("saturation", AttrOpt Float), '("pca_noise", AttrOpt Float),
        '("random_h", AttrOpt Int), '("random_s", AttrOpt Int),
        '("random_l", AttrOpt Int), '("rotate", AttrOpt Int),
        '("fill_value", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("inter_method", AttrOpt Int), '("pad", AttrOpt Int)]

_ImageRecordUInt8Iter ::
                      forall r .
                        (FieldsAcc ParameterList_ImageRecordUInt8Iter r, HasCallStack) =>
                        Record r -> IO DataIterHandle
_ImageRecordUInt8Iter args
  = let fullArgs
          = paramListWithDefault
              (Proxy @(ParameterList_ImageRecordUInt8Iter))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("path_imgidx",) . showValue <$> Anon.get #path_imgidx fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("device_id",) . showValue <$> Anon.get #device_id fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("seed_aug",) . showValue <$> Anon.get #seed_aug fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop",) . showValue <$> Anon.get #rand_crop fullArgs,
               ("random_resized_crop",) . showValue <$>
                 Anon.get #random_resized_crop fullArgs,
               ("max_rotate_angle",) . showValue <$>
                 Anon.get #max_rotate_angle fullArgs,
               ("max_aspect_ratio",) . showValue <$>
                 Anon.get #max_aspect_ratio fullArgs,
               ("min_aspect_ratio",) . showValue <$>
                 Anon.get #min_aspect_ratio fullArgs,
               ("max_shear_ratio",) . showValue <$>
                 Anon.get #max_shear_ratio fullArgs,
               ("max_crop_size",) . showValue <$>
                 Anon.get #max_crop_size fullArgs,
               ("min_crop_size",) . showValue <$>
                 Anon.get #min_crop_size fullArgs,
               ("max_random_scale",) . showValue <$>
                 Anon.get #max_random_scale fullArgs,
               ("min_random_scale",) . showValue <$>
                 Anon.get #min_random_scale fullArgs,
               ("max_random_area",) . showValue <$>
                 Anon.get #max_random_area fullArgs,
               ("min_random_area",) . showValue <$>
                 Anon.get #min_random_area fullArgs,
               ("max_img_size",) . showValue <$> Anon.get #max_img_size fullArgs,
               ("min_img_size",) . showValue <$> Anon.get #min_img_size fullArgs,
               ("brightness",) . showValue <$> Anon.get #brightness fullArgs,
               ("contrast",) . showValue <$> Anon.get #contrast fullArgs,
               ("saturation",) . showValue <$> Anon.get #saturation fullArgs,
               ("pca_noise",) . showValue <$> Anon.get #pca_noise fullArgs,
               ("random_h",) . showValue <$> Anon.get #random_h fullArgs,
               ("random_s",) . showValue <$> Anon.get #random_s fullArgs,
               ("random_l",) . showValue <$> Anon.get #random_l fullArgs,
               ("rotate",) . showValue <$> Anon.get #rotate fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 5)
         mxDataIterCreateIter di keys vals

type ParameterList_ImageRecordInt8Iter =
     '[ '("path_imglist", AttrOpt Text), '("path_imgrec", AttrOpt Text),
        '("path_imgidx", AttrOpt Text), '("aug_seq", AttrOpt Text),
        '("label_width", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("preprocess_threads", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("device_id", AttrOpt Int), '("shuffle_chunk_size", AttrOpt Int),
        '("shuffle_chunk_seed", AttrOpt Int),
        '("seed_aug", AttrOpt (Maybe Int)), '("shuffle", AttrOpt Bool),
        '("seed", AttrOpt Int), '("verbose", AttrOpt Bool),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"]))),
        '("resize", AttrOpt Int), '("rand_crop", AttrOpt Bool),
        '("random_resized_crop", AttrOpt Bool),
        '("max_rotate_angle", AttrOpt Int),
        '("max_aspect_ratio", AttrOpt Float),
        '("min_aspect_ratio", AttrOpt (Maybe Float)),
        '("max_shear_ratio", AttrOpt Float),
        '("max_crop_size", AttrOpt Int), '("min_crop_size", AttrOpt Int),
        '("max_random_scale", AttrOpt Float),
        '("min_random_scale", AttrOpt Float),
        '("max_random_area", AttrOpt Float),
        '("min_random_area", AttrOpt Float),
        '("max_img_size", AttrOpt Float), '("min_img_size", AttrOpt Float),
        '("brightness", AttrOpt Float), '("contrast", AttrOpt Float),
        '("saturation", AttrOpt Float), '("pca_noise", AttrOpt Float),
        '("random_h", AttrOpt Int), '("random_s", AttrOpt Int),
        '("random_l", AttrOpt Int), '("rotate", AttrOpt Int),
        '("fill_value", AttrOpt Int), '("data_shape", AttrReq [Int]),
        '("inter_method", AttrOpt Int), '("pad", AttrOpt Int)]

_ImageRecordInt8Iter ::
                     forall r .
                       (FieldsAcc ParameterList_ImageRecordInt8Iter r, HasCallStack) =>
                       Record r -> IO DataIterHandle
_ImageRecordInt8Iter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_ImageRecordInt8Iter))
              args
        scalarArgs
          = catMaybes
              [("path_imglist",) . showValue <$> Anon.get #path_imglist fullArgs,
               ("path_imgrec",) . showValue <$> Anon.get #path_imgrec fullArgs,
               ("path_imgidx",) . showValue <$> Anon.get #path_imgidx fullArgs,
               ("aug_seq",) . showValue <$> Anon.get #aug_seq fullArgs,
               ("label_width",) . showValue <$> Anon.get #label_width fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("preprocess_threads",) . showValue <$>
                 Anon.get #preprocess_threads fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("device_id",) . showValue <$> Anon.get #device_id fullArgs,
               ("shuffle_chunk_size",) . showValue <$>
                 Anon.get #shuffle_chunk_size fullArgs,
               ("shuffle_chunk_seed",) . showValue <$>
                 Anon.get #shuffle_chunk_seed fullArgs,
               ("seed_aug",) . showValue <$> Anon.get #seed_aug fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("verbose",) . showValue <$> Anon.get #verbose fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs,
               ("resize",) . showValue <$> Anon.get #resize fullArgs,
               ("rand_crop",) . showValue <$> Anon.get #rand_crop fullArgs,
               ("random_resized_crop",) . showValue <$>
                 Anon.get #random_resized_crop fullArgs,
               ("max_rotate_angle",) . showValue <$>
                 Anon.get #max_rotate_angle fullArgs,
               ("max_aspect_ratio",) . showValue <$>
                 Anon.get #max_aspect_ratio fullArgs,
               ("min_aspect_ratio",) . showValue <$>
                 Anon.get #min_aspect_ratio fullArgs,
               ("max_shear_ratio",) . showValue <$>
                 Anon.get #max_shear_ratio fullArgs,
               ("max_crop_size",) . showValue <$>
                 Anon.get #max_crop_size fullArgs,
               ("min_crop_size",) . showValue <$>
                 Anon.get #min_crop_size fullArgs,
               ("max_random_scale",) . showValue <$>
                 Anon.get #max_random_scale fullArgs,
               ("min_random_scale",) . showValue <$>
                 Anon.get #min_random_scale fullArgs,
               ("max_random_area",) . showValue <$>
                 Anon.get #max_random_area fullArgs,
               ("min_random_area",) . showValue <$>
                 Anon.get #min_random_area fullArgs,
               ("max_img_size",) . showValue <$> Anon.get #max_img_size fullArgs,
               ("min_img_size",) . showValue <$> Anon.get #min_img_size fullArgs,
               ("brightness",) . showValue <$> Anon.get #brightness fullArgs,
               ("contrast",) . showValue <$> Anon.get #contrast fullArgs,
               ("saturation",) . showValue <$> Anon.get #saturation fullArgs,
               ("pca_noise",) . showValue <$> Anon.get #pca_noise fullArgs,
               ("random_h",) . showValue <$> Anon.get #random_h fullArgs,
               ("random_s",) . showValue <$> Anon.get #random_s fullArgs,
               ("random_l",) . showValue <$> Anon.get #random_l fullArgs,
               ("rotate",) . showValue <$> Anon.get #rotate fullArgs,
               ("fill_value",) . showValue <$> Anon.get #fill_value fullArgs,
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("inter_method",) . showValue <$> Anon.get #inter_method fullArgs,
               ("pad",) . showValue <$> Anon.get #pad fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 6)
         mxDataIterCreateIter di keys vals

type ParameterList_LibSVMIter =
     '[ '("data_libsvm", AttrReq Text), '("data_shape", AttrReq [Int]),
        '("label_libsvm", AttrOpt Text), '("label_shape", AttrOpt [Int]),
        '("num_parts", AttrOpt Int), '("part_index", AttrOpt Int),
        '("batch_size", AttrReq Int), '("round_batch", AttrOpt Bool),
        '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"])))]

_LibSVMIter ::
            forall r . (FieldsAcc ParameterList_LibSVMIter r, HasCallStack) =>
              Record r -> IO DataIterHandle
_LibSVMIter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_LibSVMIter)) args
        scalarArgs
          = catMaybes
              [("data_libsvm",) . showValue <$>
                 Just (Anon.get #data_libsvm fullArgs),
               ("data_shape",) . showValue <$>
                 Just (Anon.get #data_shape fullArgs),
               ("label_libsvm",) . showValue <$> Anon.get #label_libsvm fullArgs,
               ("label_shape",) . showValue <$> Anon.get #label_shape fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("batch_size",) . showValue <$>
                 Just (Anon.get #batch_size fullArgs),
               ("round_batch",) . showValue <$> Anon.get #round_batch fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 7)
         mxDataIterCreateIter di keys vals

type ParameterList_MNISTIter =
     '[ '("image", AttrOpt Text), '("label", AttrOpt Text),
        '("batch_size", AttrOpt Int), '("shuffle", AttrOpt Bool),
        '("flat", AttrOpt Bool), '("seed", AttrOpt Int),
        '("silent", AttrOpt Bool), '("num_parts", AttrOpt Int),
        '("part_index", AttrOpt Int), '("prefetch_buffer", AttrOpt Int),
        '("ctx", AttrOpt (EnumType '["cpu", "gpu"])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                    "int8", "uint8"])))]

_MNISTIter ::
           forall r . (FieldsAcc ParameterList_MNISTIter r, HasCallStack) =>
             Record r -> IO DataIterHandle
_MNISTIter args
  = let fullArgs
          = paramListWithDefault (Proxy @(ParameterList_MNISTIter)) args
        scalarArgs
          = catMaybes
              [("image",) . showValue <$> Anon.get #image fullArgs,
               ("label",) . showValue <$> Anon.get #label fullArgs,
               ("batch_size",) . showValue <$> Anon.get #batch_size fullArgs,
               ("shuffle",) . showValue <$> Anon.get #shuffle fullArgs,
               ("flat",) . showValue <$> Anon.get #flat fullArgs,
               ("seed",) . showValue <$> Anon.get #seed fullArgs,
               ("silent",) . showValue <$> Anon.get #silent fullArgs,
               ("num_parts",) . showValue <$> Anon.get #num_parts fullArgs,
               ("part_index",) . showValue <$> Anon.get #part_index fullArgs,
               ("prefetch_buffer",) . showValue <$>
                 Anon.get #prefetch_buffer fullArgs,
               ("ctx",) . showValue <$> Anon.get #ctx fullArgs,
               ("dtype",) . showValue <$> Anon.get #dtype fullArgs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 8)
         mxDataIterCreateIter di keys vals