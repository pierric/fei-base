module MXNet.Base.DataIter where
import RIO
import RIO.List
import RIO.List.Partial ((!!))
import MXNet.Base.Types (DType)
import MXNet.Base.Raw
import MXNet.Base.Core.Spec
import MXNet.Base.Core.Enum
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
         forall t r .
           (Tensor t, FieldsAcc ParameterList_CSVIter r, HasCallStack) =>
           Record r -> TensorApply (IO DataIterHandle)
_CSVIter args
  = let fullArgs
          = ANON{data_csv = undefined, data_shape = undefined,
                 label_csv = Nothing, label_shape = Nothing, batch_size = undefined,
                 round_batch = Nothing, prefetch_buffer = Nothing, ctx = Nothing,
                 dtype = Nothing}
              :: ParamListFull ParameterList_CSVIter
        scalarArgs
          = [("data_csv",) . showValue $ Anon.get #data_csv fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("label_csv",) . showValue $ Anon.get #label_csv fullargs,
             ("label_shape",) . showValue $ Anon.get #label_shape fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs]
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
                    forall t r .
                      (Tensor t, FieldsAcc ParameterList_ImageDetRecordIter r,
                       HasCallStack) =>
                      Record r -> TensorApply (IO DataIterHandle)
_ImageDetRecordIter args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 aug_seq = Nothing, label_width = Nothing, data_shape = undefined,
                 preprocess_threads = Nothing, verbose = Nothing,
                 num_parts = Nothing, part_index = Nothing,
                 shuffle_chunk_size = Nothing, shuffle_chunk_seed = Nothing,
                 label_pad_width = Nothing, label_pad_value = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop_prob = Nothing,
                 min_crop_scales = Nothing, max_crop_scales = Nothing,
                 min_crop_aspect_ratios = Nothing, max_crop_aspect_ratios = Nothing,
                 min_crop_overlaps = Nothing, max_crop_overlaps = Nothing,
                 min_crop_sample_coverages = Nothing,
                 max_crop_sample_coverages = Nothing,
                 min_crop_object_coverages = Nothing,
                 max_crop_object_coverages = Nothing, num_crop_sampler = Nothing,
                 crop_emit_mode = Nothing, emit_overlap_thresh = Nothing,
                 max_crop_trials = Nothing, rand_pad_prob = Nothing,
                 max_pad_scale = Nothing, max_random_hue = Nothing,
                 random_hue_prob = Nothing, max_random_saturation = Nothing,
                 random_saturation_prob = Nothing,
                 max_random_illumination = Nothing,
                 random_illumination_prob = Nothing, max_random_contrast = Nothing,
                 random_contrast_prob = Nothing, rand_mirror_prob = Nothing,
                 fill_value = Nothing, inter_method = Nothing,
                 data_shape = undefined, resize_mode = Nothing, seed = Nothing,
                 mean_img = Nothing, mean_r = Nothing, mean_g = Nothing,
                 mean_b = Nothing, mean_a = Nothing, std_r = Nothing,
                 std_g = Nothing, std_b = Nothing, std_a = Nothing, scale = Nothing,
                 verbose = Nothing}
              :: ParamListFull ParameterList_ImageDetRecordIter
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("label_pad_width",) . showValue $
               Anon.get #label_pad_width fullargs,
             ("label_pad_value",) . showValue $
               Anon.get #label_pad_value fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop_prob",) . showValue $
               Anon.get #rand_crop_prob fullargs,
             ("min_crop_scales",) . showValue $
               Anon.get #min_crop_scales fullargs,
             ("max_crop_scales",) . showValue $
               Anon.get #max_crop_scales fullargs,
             ("min_crop_aspect_ratios",) . showValue $
               Anon.get #min_crop_aspect_ratios fullargs,
             ("max_crop_aspect_ratios",) . showValue $
               Anon.get #max_crop_aspect_ratios fullargs,
             ("min_crop_overlaps",) . showValue $
               Anon.get #min_crop_overlaps fullargs,
             ("max_crop_overlaps",) . showValue $
               Anon.get #max_crop_overlaps fullargs,
             ("min_crop_sample_coverages",) . showValue $
               Anon.get #min_crop_sample_coverages fullargs,
             ("max_crop_sample_coverages",) . showValue $
               Anon.get #max_crop_sample_coverages fullargs,
             ("min_crop_object_coverages",) . showValue $
               Anon.get #min_crop_object_coverages fullargs,
             ("max_crop_object_coverages",) . showValue $
               Anon.get #max_crop_object_coverages fullargs,
             ("num_crop_sampler",) . showValue $
               Anon.get #num_crop_sampler fullargs,
             ("crop_emit_mode",) . showValue $
               Anon.get #crop_emit_mode fullargs,
             ("emit_overlap_thresh",) . showValue $
               Anon.get #emit_overlap_thresh fullargs,
             ("max_crop_trials",) . showValue $
               Anon.get #max_crop_trials fullargs,
             ("rand_pad_prob",) . showValue $ Anon.get #rand_pad_prob fullargs,
             ("max_pad_scale",) . showValue $ Anon.get #max_pad_scale fullargs,
             ("max_random_hue",) . showValue $
               Anon.get #max_random_hue fullargs,
             ("random_hue_prob",) . showValue $
               Anon.get #random_hue_prob fullargs,
             ("max_random_saturation",) . showValue $
               Anon.get #max_random_saturation fullargs,
             ("random_saturation_prob",) . showValue $
               Anon.get #random_saturation_prob fullargs,
             ("max_random_illumination",) . showValue $
               Anon.get #max_random_illumination fullargs,
             ("random_illumination_prob",) . showValue $
               Anon.get #random_illumination_prob fullargs,
             ("max_random_contrast",) . showValue $
               Anon.get #max_random_contrast fullargs,
             ("random_contrast_prob",) . showValue $
               Anon.get #random_contrast_prob fullargs,
             ("rand_mirror_prob",) . showValue $
               Anon.get #rand_mirror_prob fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("resize_mode",) . showValue $ Anon.get #resize_mode fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("mean_img",) . showValue $ Anon.get #mean_img fullargs,
             ("mean_r",) . showValue $ Anon.get #mean_r fullargs,
             ("mean_g",) . showValue $ Anon.get #mean_g fullargs,
             ("mean_b",) . showValue $ Anon.get #mean_b fullargs,
             ("mean_a",) . showValue $ Anon.get #mean_a fullargs,
             ("std_r",) . showValue $ Anon.get #std_r fullargs,
             ("std_g",) . showValue $ Anon.get #std_g fullargs,
             ("std_b",) . showValue $ Anon.get #std_b fullargs,
             ("std_a",) . showValue $ Anon.get #std_a fullargs,
             ("scale",) . showValue $ Anon.get #scale fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs]
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
                    forall t r .
                      (Tensor t, FieldsAcc ParameterList_ImageRecordIter_v1 r,
                       HasCallStack) =>
                      Record r -> TensorApply (IO DataIterHandle)
_ImageRecordIter_v1 args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 path_imgidx = Nothing, aug_seq = Nothing, label_width = Nothing,
                 data_shape = undefined, preprocess_threads = Nothing,
                 verbose = Nothing, num_parts = Nothing, part_index = Nothing,
                 device_id = Nothing, shuffle_chunk_size = Nothing,
                 shuffle_chunk_seed = Nothing, seed_aug = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop = Nothing,
                 random_resized_crop = Nothing, max_rotate_angle = Nothing,
                 max_aspect_ratio = Nothing, min_aspect_ratio = Nothing,
                 max_shear_ratio = Nothing, max_crop_size = Nothing,
                 min_crop_size = Nothing, max_random_scale = Nothing,
                 min_random_scale = Nothing, max_random_area = Nothing,
                 min_random_area = Nothing, max_img_size = Nothing,
                 min_img_size = Nothing, brightness = Nothing, contrast = Nothing,
                 saturation = Nothing, pca_noise = Nothing, random_h = Nothing,
                 random_s = Nothing, random_l = Nothing, rotate = Nothing,
                 fill_value = Nothing, data_shape = undefined,
                 inter_method = Nothing, pad = Nothing, seed = Nothing,
                 mirror = Nothing, rand_mirror = Nothing, mean_img = Nothing,
                 mean_r = Nothing, mean_g = Nothing, mean_b = Nothing,
                 mean_a = Nothing, std_r = Nothing, std_g = Nothing,
                 std_b = Nothing, std_a = Nothing, scale = Nothing,
                 max_random_contrast = Nothing, max_random_illumination = Nothing,
                 verbose = Nothing}
              :: ParamListFull ParameterList_ImageRecordIter_v1
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("path_imgidx",) . showValue $ Anon.get #path_imgidx fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("device_id",) . showValue $ Anon.get #device_id fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("seed_aug",) . showValue $ Anon.get #seed_aug fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop",) . showValue $ Anon.get #rand_crop fullargs,
             ("random_resized_crop",) . showValue $
               Anon.get #random_resized_crop fullargs,
             ("max_rotate_angle",) . showValue $
               Anon.get #max_rotate_angle fullargs,
             ("max_aspect_ratio",) . showValue $
               Anon.get #max_aspect_ratio fullargs,
             ("min_aspect_ratio",) . showValue $
               Anon.get #min_aspect_ratio fullargs,
             ("max_shear_ratio",) . showValue $
               Anon.get #max_shear_ratio fullargs,
             ("max_crop_size",) . showValue $ Anon.get #max_crop_size fullargs,
             ("min_crop_size",) . showValue $ Anon.get #min_crop_size fullargs,
             ("max_random_scale",) . showValue $
               Anon.get #max_random_scale fullargs,
             ("min_random_scale",) . showValue $
               Anon.get #min_random_scale fullargs,
             ("max_random_area",) . showValue $
               Anon.get #max_random_area fullargs,
             ("min_random_area",) . showValue $
               Anon.get #min_random_area fullargs,
             ("max_img_size",) . showValue $ Anon.get #max_img_size fullargs,
             ("min_img_size",) . showValue $ Anon.get #min_img_size fullargs,
             ("brightness",) . showValue $ Anon.get #brightness fullargs,
             ("contrast",) . showValue $ Anon.get #contrast fullargs,
             ("saturation",) . showValue $ Anon.get #saturation fullargs,
             ("pca_noise",) . showValue $ Anon.get #pca_noise fullargs,
             ("random_h",) . showValue $ Anon.get #random_h fullargs,
             ("random_s",) . showValue $ Anon.get #random_s fullargs,
             ("random_l",) . showValue $ Anon.get #random_l fullargs,
             ("rotate",) . showValue $ Anon.get #rotate fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("pad",) . showValue $ Anon.get #pad fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("mirror",) . showValue $ Anon.get #mirror fullargs,
             ("rand_mirror",) . showValue $ Anon.get #rand_mirror fullargs,
             ("mean_img",) . showValue $ Anon.get #mean_img fullargs,
             ("mean_r",) . showValue $ Anon.get #mean_r fullargs,
             ("mean_g",) . showValue $ Anon.get #mean_g fullargs,
             ("mean_b",) . showValue $ Anon.get #mean_b fullargs,
             ("mean_a",) . showValue $ Anon.get #mean_a fullargs,
             ("std_r",) . showValue $ Anon.get #std_r fullargs,
             ("std_g",) . showValue $ Anon.get #std_g fullargs,
             ("std_b",) . showValue $ Anon.get #std_b fullargs,
             ("std_a",) . showValue $ Anon.get #std_a fullargs,
             ("scale",) . showValue $ Anon.get #scale fullargs,
             ("max_random_contrast",) . showValue $
               Anon.get #max_random_contrast fullargs,
             ("max_random_illumination",) . showValue $
               Anon.get #max_random_illumination fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs]
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
                         forall t r .
                           (Tensor t, FieldsAcc ParameterList_ImageRecordUInt8Iter_v1 r,
                            HasCallStack) =>
                           Record r -> TensorApply (IO DataIterHandle)
_ImageRecordUInt8Iter_v1 args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 path_imgidx = Nothing, aug_seq = Nothing, label_width = Nothing,
                 data_shape = undefined, preprocess_threads = Nothing,
                 verbose = Nothing, num_parts = Nothing, part_index = Nothing,
                 device_id = Nothing, shuffle_chunk_size = Nothing,
                 shuffle_chunk_seed = Nothing, seed_aug = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop = Nothing,
                 random_resized_crop = Nothing, max_rotate_angle = Nothing,
                 max_aspect_ratio = Nothing, min_aspect_ratio = Nothing,
                 max_shear_ratio = Nothing, max_crop_size = Nothing,
                 min_crop_size = Nothing, max_random_scale = Nothing,
                 min_random_scale = Nothing, max_random_area = Nothing,
                 min_random_area = Nothing, max_img_size = Nothing,
                 min_img_size = Nothing, brightness = Nothing, contrast = Nothing,
                 saturation = Nothing, pca_noise = Nothing, random_h = Nothing,
                 random_s = Nothing, random_l = Nothing, rotate = Nothing,
                 fill_value = Nothing, data_shape = undefined,
                 inter_method = Nothing, pad = Nothing}
              :: ParamListFull ParameterList_ImageRecordUInt8Iter_v1
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("path_imgidx",) . showValue $ Anon.get #path_imgidx fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("device_id",) . showValue $ Anon.get #device_id fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("seed_aug",) . showValue $ Anon.get #seed_aug fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop",) . showValue $ Anon.get #rand_crop fullargs,
             ("random_resized_crop",) . showValue $
               Anon.get #random_resized_crop fullargs,
             ("max_rotate_angle",) . showValue $
               Anon.get #max_rotate_angle fullargs,
             ("max_aspect_ratio",) . showValue $
               Anon.get #max_aspect_ratio fullargs,
             ("min_aspect_ratio",) . showValue $
               Anon.get #min_aspect_ratio fullargs,
             ("max_shear_ratio",) . showValue $
               Anon.get #max_shear_ratio fullargs,
             ("max_crop_size",) . showValue $ Anon.get #max_crop_size fullargs,
             ("min_crop_size",) . showValue $ Anon.get #min_crop_size fullargs,
             ("max_random_scale",) . showValue $
               Anon.get #max_random_scale fullargs,
             ("min_random_scale",) . showValue $
               Anon.get #min_random_scale fullargs,
             ("max_random_area",) . showValue $
               Anon.get #max_random_area fullargs,
             ("min_random_area",) . showValue $
               Anon.get #min_random_area fullargs,
             ("max_img_size",) . showValue $ Anon.get #max_img_size fullargs,
             ("min_img_size",) . showValue $ Anon.get #min_img_size fullargs,
             ("brightness",) . showValue $ Anon.get #brightness fullargs,
             ("contrast",) . showValue $ Anon.get #contrast fullargs,
             ("saturation",) . showValue $ Anon.get #saturation fullargs,
             ("pca_noise",) . showValue $ Anon.get #pca_noise fullargs,
             ("random_h",) . showValue $ Anon.get #random_h fullargs,
             ("random_s",) . showValue $ Anon.get #random_s fullargs,
             ("random_l",) . showValue $ Anon.get #random_l fullargs,
             ("rotate",) . showValue $ Anon.get #rotate fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("pad",) . showValue $ Anon.get #pad fullargs]
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
                 forall t r .
                   (Tensor t, FieldsAcc ParameterList_ImageRecordIter r,
                    HasCallStack) =>
                   Record r -> TensorApply (IO DataIterHandle)
_ImageRecordIter args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 path_imgidx = Nothing, aug_seq = Nothing, label_width = Nothing,
                 data_shape = undefined, preprocess_threads = Nothing,
                 verbose = Nothing, num_parts = Nothing, part_index = Nothing,
                 device_id = Nothing, shuffle_chunk_size = Nothing,
                 shuffle_chunk_seed = Nothing, seed_aug = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop = Nothing,
                 random_resized_crop = Nothing, max_rotate_angle = Nothing,
                 max_aspect_ratio = Nothing, min_aspect_ratio = Nothing,
                 max_shear_ratio = Nothing, max_crop_size = Nothing,
                 min_crop_size = Nothing, max_random_scale = Nothing,
                 min_random_scale = Nothing, max_random_area = Nothing,
                 min_random_area = Nothing, max_img_size = Nothing,
                 min_img_size = Nothing, brightness = Nothing, contrast = Nothing,
                 saturation = Nothing, pca_noise = Nothing, random_h = Nothing,
                 random_s = Nothing, random_l = Nothing, rotate = Nothing,
                 fill_value = Nothing, data_shape = undefined,
                 inter_method = Nothing, pad = Nothing, seed = Nothing,
                 mirror = Nothing, rand_mirror = Nothing, mean_img = Nothing,
                 mean_r = Nothing, mean_g = Nothing, mean_b = Nothing,
                 mean_a = Nothing, std_r = Nothing, std_g = Nothing,
                 std_b = Nothing, std_a = Nothing, scale = Nothing,
                 max_random_contrast = Nothing, max_random_illumination = Nothing,
                 verbose = Nothing}
              :: ParamListFull ParameterList_ImageRecordIter
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("path_imgidx",) . showValue $ Anon.get #path_imgidx fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("device_id",) . showValue $ Anon.get #device_id fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("seed_aug",) . showValue $ Anon.get #seed_aug fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop",) . showValue $ Anon.get #rand_crop fullargs,
             ("random_resized_crop",) . showValue $
               Anon.get #random_resized_crop fullargs,
             ("max_rotate_angle",) . showValue $
               Anon.get #max_rotate_angle fullargs,
             ("max_aspect_ratio",) . showValue $
               Anon.get #max_aspect_ratio fullargs,
             ("min_aspect_ratio",) . showValue $
               Anon.get #min_aspect_ratio fullargs,
             ("max_shear_ratio",) . showValue $
               Anon.get #max_shear_ratio fullargs,
             ("max_crop_size",) . showValue $ Anon.get #max_crop_size fullargs,
             ("min_crop_size",) . showValue $ Anon.get #min_crop_size fullargs,
             ("max_random_scale",) . showValue $
               Anon.get #max_random_scale fullargs,
             ("min_random_scale",) . showValue $
               Anon.get #min_random_scale fullargs,
             ("max_random_area",) . showValue $
               Anon.get #max_random_area fullargs,
             ("min_random_area",) . showValue $
               Anon.get #min_random_area fullargs,
             ("max_img_size",) . showValue $ Anon.get #max_img_size fullargs,
             ("min_img_size",) . showValue $ Anon.get #min_img_size fullargs,
             ("brightness",) . showValue $ Anon.get #brightness fullargs,
             ("contrast",) . showValue $ Anon.get #contrast fullargs,
             ("saturation",) . showValue $ Anon.get #saturation fullargs,
             ("pca_noise",) . showValue $ Anon.get #pca_noise fullargs,
             ("random_h",) . showValue $ Anon.get #random_h fullargs,
             ("random_s",) . showValue $ Anon.get #random_s fullargs,
             ("random_l",) . showValue $ Anon.get #random_l fullargs,
             ("rotate",) . showValue $ Anon.get #rotate fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("pad",) . showValue $ Anon.get #pad fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("mirror",) . showValue $ Anon.get #mirror fullargs,
             ("rand_mirror",) . showValue $ Anon.get #rand_mirror fullargs,
             ("mean_img",) . showValue $ Anon.get #mean_img fullargs,
             ("mean_r",) . showValue $ Anon.get #mean_r fullargs,
             ("mean_g",) . showValue $ Anon.get #mean_g fullargs,
             ("mean_b",) . showValue $ Anon.get #mean_b fullargs,
             ("mean_a",) . showValue $ Anon.get #mean_a fullargs,
             ("std_r",) . showValue $ Anon.get #std_r fullargs,
             ("std_g",) . showValue $ Anon.get #std_g fullargs,
             ("std_b",) . showValue $ Anon.get #std_b fullargs,
             ("std_a",) . showValue $ Anon.get #std_a fullargs,
             ("scale",) . showValue $ Anon.get #scale fullargs,
             ("max_random_contrast",) . showValue $
               Anon.get #max_random_contrast fullargs,
             ("max_random_illumination",) . showValue $
               Anon.get #max_random_illumination fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs]
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
                      forall t r .
                        (Tensor t, FieldsAcc ParameterList_ImageRecordUInt8Iter r,
                         HasCallStack) =>
                        Record r -> TensorApply (IO DataIterHandle)
_ImageRecordUInt8Iter args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 path_imgidx = Nothing, aug_seq = Nothing, label_width = Nothing,
                 data_shape = undefined, preprocess_threads = Nothing,
                 verbose = Nothing, num_parts = Nothing, part_index = Nothing,
                 device_id = Nothing, shuffle_chunk_size = Nothing,
                 shuffle_chunk_seed = Nothing, seed_aug = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop = Nothing,
                 random_resized_crop = Nothing, max_rotate_angle = Nothing,
                 max_aspect_ratio = Nothing, min_aspect_ratio = Nothing,
                 max_shear_ratio = Nothing, max_crop_size = Nothing,
                 min_crop_size = Nothing, max_random_scale = Nothing,
                 min_random_scale = Nothing, max_random_area = Nothing,
                 min_random_area = Nothing, max_img_size = Nothing,
                 min_img_size = Nothing, brightness = Nothing, contrast = Nothing,
                 saturation = Nothing, pca_noise = Nothing, random_h = Nothing,
                 random_s = Nothing, random_l = Nothing, rotate = Nothing,
                 fill_value = Nothing, data_shape = undefined,
                 inter_method = Nothing, pad = Nothing}
              :: ParamListFull ParameterList_ImageRecordUInt8Iter
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("path_imgidx",) . showValue $ Anon.get #path_imgidx fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("device_id",) . showValue $ Anon.get #device_id fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("seed_aug",) . showValue $ Anon.get #seed_aug fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop",) . showValue $ Anon.get #rand_crop fullargs,
             ("random_resized_crop",) . showValue $
               Anon.get #random_resized_crop fullargs,
             ("max_rotate_angle",) . showValue $
               Anon.get #max_rotate_angle fullargs,
             ("max_aspect_ratio",) . showValue $
               Anon.get #max_aspect_ratio fullargs,
             ("min_aspect_ratio",) . showValue $
               Anon.get #min_aspect_ratio fullargs,
             ("max_shear_ratio",) . showValue $
               Anon.get #max_shear_ratio fullargs,
             ("max_crop_size",) . showValue $ Anon.get #max_crop_size fullargs,
             ("min_crop_size",) . showValue $ Anon.get #min_crop_size fullargs,
             ("max_random_scale",) . showValue $
               Anon.get #max_random_scale fullargs,
             ("min_random_scale",) . showValue $
               Anon.get #min_random_scale fullargs,
             ("max_random_area",) . showValue $
               Anon.get #max_random_area fullargs,
             ("min_random_area",) . showValue $
               Anon.get #min_random_area fullargs,
             ("max_img_size",) . showValue $ Anon.get #max_img_size fullargs,
             ("min_img_size",) . showValue $ Anon.get #min_img_size fullargs,
             ("brightness",) . showValue $ Anon.get #brightness fullargs,
             ("contrast",) . showValue $ Anon.get #contrast fullargs,
             ("saturation",) . showValue $ Anon.get #saturation fullargs,
             ("pca_noise",) . showValue $ Anon.get #pca_noise fullargs,
             ("random_h",) . showValue $ Anon.get #random_h fullargs,
             ("random_s",) . showValue $ Anon.get #random_s fullargs,
             ("random_l",) . showValue $ Anon.get #random_l fullargs,
             ("rotate",) . showValue $ Anon.get #rotate fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("pad",) . showValue $ Anon.get #pad fullargs]
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
                     forall t r .
                       (Tensor t, FieldsAcc ParameterList_ImageRecordInt8Iter r,
                        HasCallStack) =>
                       Record r -> TensorApply (IO DataIterHandle)
_ImageRecordInt8Iter args
  = let fullArgs
          = ANON{path_imglist = Nothing, path_imgrec = Nothing,
                 path_imgidx = Nothing, aug_seq = Nothing, label_width = Nothing,
                 data_shape = undefined, preprocess_threads = Nothing,
                 verbose = Nothing, num_parts = Nothing, part_index = Nothing,
                 device_id = Nothing, shuffle_chunk_size = Nothing,
                 shuffle_chunk_seed = Nothing, seed_aug = Nothing,
                 shuffle = Nothing, seed = Nothing, verbose = Nothing,
                 batch_size = undefined, round_batch = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing,
                 resize = Nothing, rand_crop = Nothing,
                 random_resized_crop = Nothing, max_rotate_angle = Nothing,
                 max_aspect_ratio = Nothing, min_aspect_ratio = Nothing,
                 max_shear_ratio = Nothing, max_crop_size = Nothing,
                 min_crop_size = Nothing, max_random_scale = Nothing,
                 min_random_scale = Nothing, max_random_area = Nothing,
                 min_random_area = Nothing, max_img_size = Nothing,
                 min_img_size = Nothing, brightness = Nothing, contrast = Nothing,
                 saturation = Nothing, pca_noise = Nothing, random_h = Nothing,
                 random_s = Nothing, random_l = Nothing, rotate = Nothing,
                 fill_value = Nothing, data_shape = undefined,
                 inter_method = Nothing, pad = Nothing}
              :: ParamListFull ParameterList_ImageRecordInt8Iter
        scalarArgs
          = [("path_imglist",) . showValue $ Anon.get #path_imglist fullargs,
             ("path_imgrec",) . showValue $ Anon.get #path_imgrec fullargs,
             ("path_imgidx",) . showValue $ Anon.get #path_imgidx fullargs,
             ("aug_seq",) . showValue $ Anon.get #aug_seq fullargs,
             ("label_width",) . showValue $ Anon.get #label_width fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("preprocess_threads",) . showValue $
               Anon.get #preprocess_threads fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("device_id",) . showValue $ Anon.get #device_id fullargs,
             ("shuffle_chunk_size",) . showValue $
               Anon.get #shuffle_chunk_size fullargs,
             ("shuffle_chunk_seed",) . showValue $
               Anon.get #shuffle_chunk_seed fullargs,
             ("seed_aug",) . showValue $ Anon.get #seed_aug fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("verbose",) . showValue $ Anon.get #verbose fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs,
             ("resize",) . showValue $ Anon.get #resize fullargs,
             ("rand_crop",) . showValue $ Anon.get #rand_crop fullargs,
             ("random_resized_crop",) . showValue $
               Anon.get #random_resized_crop fullargs,
             ("max_rotate_angle",) . showValue $
               Anon.get #max_rotate_angle fullargs,
             ("max_aspect_ratio",) . showValue $
               Anon.get #max_aspect_ratio fullargs,
             ("min_aspect_ratio",) . showValue $
               Anon.get #min_aspect_ratio fullargs,
             ("max_shear_ratio",) . showValue $
               Anon.get #max_shear_ratio fullargs,
             ("max_crop_size",) . showValue $ Anon.get #max_crop_size fullargs,
             ("min_crop_size",) . showValue $ Anon.get #min_crop_size fullargs,
             ("max_random_scale",) . showValue $
               Anon.get #max_random_scale fullargs,
             ("min_random_scale",) . showValue $
               Anon.get #min_random_scale fullargs,
             ("max_random_area",) . showValue $
               Anon.get #max_random_area fullargs,
             ("min_random_area",) . showValue $
               Anon.get #min_random_area fullargs,
             ("max_img_size",) . showValue $ Anon.get #max_img_size fullargs,
             ("min_img_size",) . showValue $ Anon.get #min_img_size fullargs,
             ("brightness",) . showValue $ Anon.get #brightness fullargs,
             ("contrast",) . showValue $ Anon.get #contrast fullargs,
             ("saturation",) . showValue $ Anon.get #saturation fullargs,
             ("pca_noise",) . showValue $ Anon.get #pca_noise fullargs,
             ("random_h",) . showValue $ Anon.get #random_h fullargs,
             ("random_s",) . showValue $ Anon.get #random_s fullargs,
             ("random_l",) . showValue $ Anon.get #random_l fullargs,
             ("rotate",) . showValue $ Anon.get #rotate fullargs,
             ("fill_value",) . showValue $ Anon.get #fill_value fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("inter_method",) . showValue $ Anon.get #inter_method fullargs,
             ("pad",) . showValue $ Anon.get #pad fullargs]
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
            forall t r .
              (Tensor t, FieldsAcc ParameterList_LibSVMIter r, HasCallStack) =>
              Record r -> TensorApply (IO DataIterHandle)
_LibSVMIter args
  = let fullArgs
          = ANON{data_libsvm = undefined, data_shape = undefined,
                 label_libsvm = Nothing, label_shape = Nothing, num_parts = Nothing,
                 part_index = Nothing, batch_size = undefined,
                 round_batch = Nothing, prefetch_buffer = Nothing, ctx = Nothing,
                 dtype = Nothing}
              :: ParamListFull ParameterList_LibSVMIter
        scalarArgs
          = [("data_libsvm",) . showValue $ Anon.get #data_libsvm fullargs,
             ("data_shape",) . showValue $ Anon.get #data_shape fullargs,
             ("label_libsvm",) . showValue $ Anon.get #label_libsvm fullargs,
             ("label_shape",) . showValue $ Anon.get #label_shape fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("round_batch",) . showValue $ Anon.get #round_batch fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs]
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
           forall t r .
             (Tensor t, FieldsAcc ParameterList_MNISTIter r, HasCallStack) =>
             Record r -> TensorApply (IO DataIterHandle)
_MNISTIter args
  = let fullArgs
          = ANON{image = Nothing, label = Nothing, batch_size = Nothing,
                 shuffle = Nothing, flat = Nothing, seed = Nothing,
                 silent = Nothing, num_parts = Nothing, part_index = Nothing,
                 prefetch_buffer = Nothing, ctx = Nothing, dtype = Nothing}
              :: ParamListFull ParameterList_MNISTIter
        scalarArgs
          = [("image",) . showValue $ Anon.get #image fullargs,
             ("label",) . showValue $ Anon.get #label fullargs,
             ("batch_size",) . showValue $ Anon.get #batch_size fullargs,
             ("shuffle",) . showValue $ Anon.get #shuffle fullargs,
             ("flat",) . showValue $ Anon.get #flat fullargs,
             ("seed",) . showValue $ Anon.get #seed fullargs,
             ("silent",) . showValue $ Anon.get #silent fullargs,
             ("num_parts",) . showValue $ Anon.get #num_parts fullargs,
             ("part_index",) . showValue $ Anon.get #part_index fullargs,
             ("prefetch_buffer",) . showValue $
               Anon.get #prefetch_buffer fullargs,
             ("ctx",) . showValue $ Anon.get #ctx fullargs,
             ("dtype",) . showValue $ Anon.get #dtype fullargs]
        (keys, vals) = unzip scalarArgs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 8)
         mxDataIterCreateIter di keys vals