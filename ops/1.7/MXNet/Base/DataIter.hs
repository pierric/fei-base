module MXNet.Base.DataIter where
import RIO
import RIO.List
import RIO.List.Partial ((!!))
import MXNet.Base.Raw
import MXNet.Base.Spec.Operator
import MXNet.Base.Spec.HMap
import Data.Maybe (catMaybes, fromMaybe)

type instance ParameterList "_CSVIter" dummy =
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
         forall a . Fullfilled "_CSVIter" () a =>
           ArgsHMap "_CSVIter" () a -> IO DataIterHandle
_CSVIter args
  = let allargs
          = catMaybes
              [("data_csv",) . showValue <$> (args !? #data_csv :: Maybe Text),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("label_csv",) . showValue <$> (args !? #label_csv :: Maybe Text),
               ("label_shape",) . showValue <$>
                 (args !? #label_shape :: Maybe [Int]),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"])))]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 0)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageDetRecordIter" dummy =
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
                    forall a . Fullfilled "_ImageDetRecordIter" () a =>
                      ArgsHMap "_ImageDetRecordIter" () a -> IO DataIterHandle
_ImageDetRecordIter args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("label_pad_width",) . showValue <$>
                 (args !? #label_pad_width :: Maybe Int),
               ("label_pad_value",) . showValue <$>
                 (args !? #label_pad_value :: Maybe Float),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop_prob",) . showValue <$>
                 (args !? #rand_crop_prob :: Maybe Float),
               ("min_crop_scales",) . showValue <$>
                 (args !? #min_crop_scales :: Maybe [Float]),
               ("max_crop_scales",) . showValue <$>
                 (args !? #max_crop_scales :: Maybe [Float]),
               ("min_crop_aspect_ratios",) . showValue <$>
                 (args !? #min_crop_aspect_ratios :: Maybe [Float]),
               ("max_crop_aspect_ratios",) . showValue <$>
                 (args !? #max_crop_aspect_ratios :: Maybe [Float]),
               ("min_crop_overlaps",) . showValue <$>
                 (args !? #min_crop_overlaps :: Maybe [Float]),
               ("max_crop_overlaps",) . showValue <$>
                 (args !? #max_crop_overlaps :: Maybe [Float]),
               ("min_crop_sample_coverages",) . showValue <$>
                 (args !? #min_crop_sample_coverages :: Maybe [Float]),
               ("max_crop_sample_coverages",) . showValue <$>
                 (args !? #max_crop_sample_coverages :: Maybe [Float]),
               ("min_crop_object_coverages",) . showValue <$>
                 (args !? #min_crop_object_coverages :: Maybe [Float]),
               ("max_crop_object_coverages",) . showValue <$>
                 (args !? #max_crop_object_coverages :: Maybe [Float]),
               ("num_crop_sampler",) . showValue <$>
                 (args !? #num_crop_sampler :: Maybe Int),
               ("crop_emit_mode",) . showValue <$>
                 (args !? #crop_emit_mode ::
                    Maybe (EnumType '["center", "overlap"])),
               ("emit_overlap_thresh",) . showValue <$>
                 (args !? #emit_overlap_thresh :: Maybe Float),
               ("max_crop_trials",) . showValue <$>
                 (args !? #max_crop_trials :: Maybe [Int]),
               ("rand_pad_prob",) . showValue <$>
                 (args !? #rand_pad_prob :: Maybe Float),
               ("max_pad_scale",) . showValue <$>
                 (args !? #max_pad_scale :: Maybe Float),
               ("max_random_hue",) . showValue <$>
                 (args !? #max_random_hue :: Maybe Int),
               ("random_hue_prob",) . showValue <$>
                 (args !? #random_hue_prob :: Maybe Float),
               ("max_random_saturation",) . showValue <$>
                 (args !? #max_random_saturation :: Maybe Int),
               ("random_saturation_prob",) . showValue <$>
                 (args !? #random_saturation_prob :: Maybe Float),
               ("max_random_illumination",) . showValue <$>
                 (args !? #max_random_illumination :: Maybe Int),
               ("random_illumination_prob",) . showValue <$>
                 (args !? #random_illumination_prob :: Maybe Float),
               ("max_random_contrast",) . showValue <$>
                 (args !? #max_random_contrast :: Maybe Float),
               ("random_contrast_prob",) . showValue <$>
                 (args !? #random_contrast_prob :: Maybe Float),
               ("rand_mirror_prob",) . showValue <$>
                 (args !? #rand_mirror_prob :: Maybe Float),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("resize_mode",) . showValue <$>
                 (args !? #resize_mode ::
                    Maybe (EnumType '["fit", "force", "shrink"])),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("mean_img",) . showValue <$> (args !? #mean_img :: Maybe Text),
               ("mean_r",) . showValue <$> (args !? #mean_r :: Maybe Float),
               ("mean_g",) . showValue <$> (args !? #mean_g :: Maybe Float),
               ("mean_b",) . showValue <$> (args !? #mean_b :: Maybe Float),
               ("mean_a",) . showValue <$> (args !? #mean_a :: Maybe Float),
               ("std_r",) . showValue <$> (args !? #std_r :: Maybe Float),
               ("std_g",) . showValue <$> (args !? #std_g :: Maybe Float),
               ("std_b",) . showValue <$> (args !? #std_b :: Maybe Float),
               ("std_a",) . showValue <$> (args !? #std_a :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 1)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageRecordIter_v1" dummy =
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
                    forall a . Fullfilled "_ImageRecordIter_v1" () a =>
                      ArgsHMap "_ImageRecordIter_v1" () a -> IO DataIterHandle
_ImageRecordIter_v1 args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("path_imgidx",) . showValue <$>
                 (args !? #path_imgidx :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("device_id",) . showValue <$> (args !? #device_id :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("seed_aug",) . showValue <$>
                 (args !? #seed_aug :: Maybe (Maybe Int)),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop",) . showValue <$> (args !? #rand_crop :: Maybe Bool),
               ("random_resized_crop",) . showValue <$>
                 (args !? #random_resized_crop :: Maybe Bool),
               ("max_rotate_angle",) . showValue <$>
                 (args !? #max_rotate_angle :: Maybe Int),
               ("max_aspect_ratio",) . showValue <$>
                 (args !? #max_aspect_ratio :: Maybe Float),
               ("min_aspect_ratio",) . showValue <$>
                 (args !? #min_aspect_ratio :: Maybe (Maybe Float)),
               ("max_shear_ratio",) . showValue <$>
                 (args !? #max_shear_ratio :: Maybe Float),
               ("max_crop_size",) . showValue <$>
                 (args !? #max_crop_size :: Maybe Int),
               ("min_crop_size",) . showValue <$>
                 (args !? #min_crop_size :: Maybe Int),
               ("max_random_scale",) . showValue <$>
                 (args !? #max_random_scale :: Maybe Float),
               ("min_random_scale",) . showValue <$>
                 (args !? #min_random_scale :: Maybe Float),
               ("max_random_area",) . showValue <$>
                 (args !? #max_random_area :: Maybe Float),
               ("min_random_area",) . showValue <$>
                 (args !? #min_random_area :: Maybe Float),
               ("max_img_size",) . showValue <$>
                 (args !? #max_img_size :: Maybe Float),
               ("min_img_size",) . showValue <$>
                 (args !? #min_img_size :: Maybe Float),
               ("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("pca_noise",) . showValue <$> (args !? #pca_noise :: Maybe Float),
               ("random_h",) . showValue <$> (args !? #random_h :: Maybe Int),
               ("random_s",) . showValue <$> (args !? #random_s :: Maybe Int),
               ("random_l",) . showValue <$> (args !? #random_l :: Maybe Int),
               ("rotate",) . showValue <$> (args !? #rotate :: Maybe Int),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("pad",) . showValue <$> (args !? #pad :: Maybe Int),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("mirror",) . showValue <$> (args !? #mirror :: Maybe Bool),
               ("rand_mirror",) . showValue <$>
                 (args !? #rand_mirror :: Maybe Bool),
               ("mean_img",) . showValue <$> (args !? #mean_img :: Maybe Text),
               ("mean_r",) . showValue <$> (args !? #mean_r :: Maybe Float),
               ("mean_g",) . showValue <$> (args !? #mean_g :: Maybe Float),
               ("mean_b",) . showValue <$> (args !? #mean_b :: Maybe Float),
               ("mean_a",) . showValue <$> (args !? #mean_a :: Maybe Float),
               ("std_r",) . showValue <$> (args !? #std_r :: Maybe Float),
               ("std_g",) . showValue <$> (args !? #std_g :: Maybe Float),
               ("std_b",) . showValue <$> (args !? #std_b :: Maybe Float),
               ("std_a",) . showValue <$> (args !? #std_a :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("max_random_contrast",) . showValue <$>
                 (args !? #max_random_contrast :: Maybe Float),
               ("max_random_illumination",) . showValue <$>
                 (args !? #max_random_illumination :: Maybe Float),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 2)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageRecordUInt8Iter_v1" dummy =
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
                         forall a . Fullfilled "_ImageRecordUInt8Iter_v1" () a =>
                           ArgsHMap "_ImageRecordUInt8Iter_v1" () a -> IO DataIterHandle
_ImageRecordUInt8Iter_v1 args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("path_imgidx",) . showValue <$>
                 (args !? #path_imgidx :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("device_id",) . showValue <$> (args !? #device_id :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("seed_aug",) . showValue <$>
                 (args !? #seed_aug :: Maybe (Maybe Int)),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop",) . showValue <$> (args !? #rand_crop :: Maybe Bool),
               ("random_resized_crop",) . showValue <$>
                 (args !? #random_resized_crop :: Maybe Bool),
               ("max_rotate_angle",) . showValue <$>
                 (args !? #max_rotate_angle :: Maybe Int),
               ("max_aspect_ratio",) . showValue <$>
                 (args !? #max_aspect_ratio :: Maybe Float),
               ("min_aspect_ratio",) . showValue <$>
                 (args !? #min_aspect_ratio :: Maybe (Maybe Float)),
               ("max_shear_ratio",) . showValue <$>
                 (args !? #max_shear_ratio :: Maybe Float),
               ("max_crop_size",) . showValue <$>
                 (args !? #max_crop_size :: Maybe Int),
               ("min_crop_size",) . showValue <$>
                 (args !? #min_crop_size :: Maybe Int),
               ("max_random_scale",) . showValue <$>
                 (args !? #max_random_scale :: Maybe Float),
               ("min_random_scale",) . showValue <$>
                 (args !? #min_random_scale :: Maybe Float),
               ("max_random_area",) . showValue <$>
                 (args !? #max_random_area :: Maybe Float),
               ("min_random_area",) . showValue <$>
                 (args !? #min_random_area :: Maybe Float),
               ("max_img_size",) . showValue <$>
                 (args !? #max_img_size :: Maybe Float),
               ("min_img_size",) . showValue <$>
                 (args !? #min_img_size :: Maybe Float),
               ("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("pca_noise",) . showValue <$> (args !? #pca_noise :: Maybe Float),
               ("random_h",) . showValue <$> (args !? #random_h :: Maybe Int),
               ("random_s",) . showValue <$> (args !? #random_s :: Maybe Int),
               ("random_l",) . showValue <$> (args !? #random_l :: Maybe Int),
               ("rotate",) . showValue <$> (args !? #rotate :: Maybe Int),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("pad",) . showValue <$> (args !? #pad :: Maybe Int)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 3)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageRecordIter" dummy =
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
                 forall a . Fullfilled "_ImageRecordIter" () a =>
                   ArgsHMap "_ImageRecordIter" () a -> IO DataIterHandle
_ImageRecordIter args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("path_imgidx",) . showValue <$>
                 (args !? #path_imgidx :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("device_id",) . showValue <$> (args !? #device_id :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("seed_aug",) . showValue <$>
                 (args !? #seed_aug :: Maybe (Maybe Int)),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop",) . showValue <$> (args !? #rand_crop :: Maybe Bool),
               ("random_resized_crop",) . showValue <$>
                 (args !? #random_resized_crop :: Maybe Bool),
               ("max_rotate_angle",) . showValue <$>
                 (args !? #max_rotate_angle :: Maybe Int),
               ("max_aspect_ratio",) . showValue <$>
                 (args !? #max_aspect_ratio :: Maybe Float),
               ("min_aspect_ratio",) . showValue <$>
                 (args !? #min_aspect_ratio :: Maybe (Maybe Float)),
               ("max_shear_ratio",) . showValue <$>
                 (args !? #max_shear_ratio :: Maybe Float),
               ("max_crop_size",) . showValue <$>
                 (args !? #max_crop_size :: Maybe Int),
               ("min_crop_size",) . showValue <$>
                 (args !? #min_crop_size :: Maybe Int),
               ("max_random_scale",) . showValue <$>
                 (args !? #max_random_scale :: Maybe Float),
               ("min_random_scale",) . showValue <$>
                 (args !? #min_random_scale :: Maybe Float),
               ("max_random_area",) . showValue <$>
                 (args !? #max_random_area :: Maybe Float),
               ("min_random_area",) . showValue <$>
                 (args !? #min_random_area :: Maybe Float),
               ("max_img_size",) . showValue <$>
                 (args !? #max_img_size :: Maybe Float),
               ("min_img_size",) . showValue <$>
                 (args !? #min_img_size :: Maybe Float),
               ("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("pca_noise",) . showValue <$> (args !? #pca_noise :: Maybe Float),
               ("random_h",) . showValue <$> (args !? #random_h :: Maybe Int),
               ("random_s",) . showValue <$> (args !? #random_s :: Maybe Int),
               ("random_l",) . showValue <$> (args !? #random_l :: Maybe Int),
               ("rotate",) . showValue <$> (args !? #rotate :: Maybe Int),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("pad",) . showValue <$> (args !? #pad :: Maybe Int),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("mirror",) . showValue <$> (args !? #mirror :: Maybe Bool),
               ("rand_mirror",) . showValue <$>
                 (args !? #rand_mirror :: Maybe Bool),
               ("mean_img",) . showValue <$> (args !? #mean_img :: Maybe Text),
               ("mean_r",) . showValue <$> (args !? #mean_r :: Maybe Float),
               ("mean_g",) . showValue <$> (args !? #mean_g :: Maybe Float),
               ("mean_b",) . showValue <$> (args !? #mean_b :: Maybe Float),
               ("mean_a",) . showValue <$> (args !? #mean_a :: Maybe Float),
               ("std_r",) . showValue <$> (args !? #std_r :: Maybe Float),
               ("std_g",) . showValue <$> (args !? #std_g :: Maybe Float),
               ("std_b",) . showValue <$> (args !? #std_b :: Maybe Float),
               ("std_a",) . showValue <$> (args !? #std_a :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("max_random_contrast",) . showValue <$>
                 (args !? #max_random_contrast :: Maybe Float),
               ("max_random_illumination",) . showValue <$>
                 (args !? #max_random_illumination :: Maybe Float),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 4)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageRecordUInt8Iter" dummy =
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
                      forall a . Fullfilled "_ImageRecordUInt8Iter" () a =>
                        ArgsHMap "_ImageRecordUInt8Iter" () a -> IO DataIterHandle
_ImageRecordUInt8Iter args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("path_imgidx",) . showValue <$>
                 (args !? #path_imgidx :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("device_id",) . showValue <$> (args !? #device_id :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("seed_aug",) . showValue <$>
                 (args !? #seed_aug :: Maybe (Maybe Int)),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop",) . showValue <$> (args !? #rand_crop :: Maybe Bool),
               ("random_resized_crop",) . showValue <$>
                 (args !? #random_resized_crop :: Maybe Bool),
               ("max_rotate_angle",) . showValue <$>
                 (args !? #max_rotate_angle :: Maybe Int),
               ("max_aspect_ratio",) . showValue <$>
                 (args !? #max_aspect_ratio :: Maybe Float),
               ("min_aspect_ratio",) . showValue <$>
                 (args !? #min_aspect_ratio :: Maybe (Maybe Float)),
               ("max_shear_ratio",) . showValue <$>
                 (args !? #max_shear_ratio :: Maybe Float),
               ("max_crop_size",) . showValue <$>
                 (args !? #max_crop_size :: Maybe Int),
               ("min_crop_size",) . showValue <$>
                 (args !? #min_crop_size :: Maybe Int),
               ("max_random_scale",) . showValue <$>
                 (args !? #max_random_scale :: Maybe Float),
               ("min_random_scale",) . showValue <$>
                 (args !? #min_random_scale :: Maybe Float),
               ("max_random_area",) . showValue <$>
                 (args !? #max_random_area :: Maybe Float),
               ("min_random_area",) . showValue <$>
                 (args !? #min_random_area :: Maybe Float),
               ("max_img_size",) . showValue <$>
                 (args !? #max_img_size :: Maybe Float),
               ("min_img_size",) . showValue <$>
                 (args !? #min_img_size :: Maybe Float),
               ("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("pca_noise",) . showValue <$> (args !? #pca_noise :: Maybe Float),
               ("random_h",) . showValue <$> (args !? #random_h :: Maybe Int),
               ("random_s",) . showValue <$> (args !? #random_s :: Maybe Int),
               ("random_l",) . showValue <$> (args !? #random_l :: Maybe Int),
               ("rotate",) . showValue <$> (args !? #rotate :: Maybe Int),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("pad",) . showValue <$> (args !? #pad :: Maybe Int)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 5)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_ImageRecordInt8Iter" dummy =
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
                     forall a . Fullfilled "_ImageRecordInt8Iter" () a =>
                       ArgsHMap "_ImageRecordInt8Iter" () a -> IO DataIterHandle
_ImageRecordInt8Iter args
  = let allargs
          = catMaybes
              [("path_imglist",) . showValue <$>
                 (args !? #path_imglist :: Maybe Text),
               ("path_imgrec",) . showValue <$>
                 (args !? #path_imgrec :: Maybe Text),
               ("path_imgidx",) . showValue <$>
                 (args !? #path_imgidx :: Maybe Text),
               ("aug_seq",) . showValue <$> (args !? #aug_seq :: Maybe Text),
               ("label_width",) . showValue <$>
                 (args !? #label_width :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("preprocess_threads",) . showValue <$>
                 (args !? #preprocess_threads :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("device_id",) . showValue <$> (args !? #device_id :: Maybe Int),
               ("shuffle_chunk_size",) . showValue <$>
                 (args !? #shuffle_chunk_size :: Maybe Int),
               ("shuffle_chunk_seed",) . showValue <$>
                 (args !? #shuffle_chunk_seed :: Maybe Int),
               ("seed_aug",) . showValue <$>
                 (args !? #seed_aug :: Maybe (Maybe Int)),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("verbose",) . showValue <$> (args !? #verbose :: Maybe Bool),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"]))),
               ("resize",) . showValue <$> (args !? #resize :: Maybe Int),
               ("rand_crop",) . showValue <$> (args !? #rand_crop :: Maybe Bool),
               ("random_resized_crop",) . showValue <$>
                 (args !? #random_resized_crop :: Maybe Bool),
               ("max_rotate_angle",) . showValue <$>
                 (args !? #max_rotate_angle :: Maybe Int),
               ("max_aspect_ratio",) . showValue <$>
                 (args !? #max_aspect_ratio :: Maybe Float),
               ("min_aspect_ratio",) . showValue <$>
                 (args !? #min_aspect_ratio :: Maybe (Maybe Float)),
               ("max_shear_ratio",) . showValue <$>
                 (args !? #max_shear_ratio :: Maybe Float),
               ("max_crop_size",) . showValue <$>
                 (args !? #max_crop_size :: Maybe Int),
               ("min_crop_size",) . showValue <$>
                 (args !? #min_crop_size :: Maybe Int),
               ("max_random_scale",) . showValue <$>
                 (args !? #max_random_scale :: Maybe Float),
               ("min_random_scale",) . showValue <$>
                 (args !? #min_random_scale :: Maybe Float),
               ("max_random_area",) . showValue <$>
                 (args !? #max_random_area :: Maybe Float),
               ("min_random_area",) . showValue <$>
                 (args !? #min_random_area :: Maybe Float),
               ("max_img_size",) . showValue <$>
                 (args !? #max_img_size :: Maybe Float),
               ("min_img_size",) . showValue <$>
                 (args !? #min_img_size :: Maybe Float),
               ("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("pca_noise",) . showValue <$> (args !? #pca_noise :: Maybe Float),
               ("random_h",) . showValue <$> (args !? #random_h :: Maybe Int),
               ("random_s",) . showValue <$> (args !? #random_s :: Maybe Int),
               ("random_l",) . showValue <$> (args !? #random_l :: Maybe Int),
               ("rotate",) . showValue <$> (args !? #rotate :: Maybe Int),
               ("fill_value",) . showValue <$> (args !? #fill_value :: Maybe Int),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("inter_method",) . showValue <$>
                 (args !? #inter_method :: Maybe Int),
               ("pad",) . showValue <$> (args !? #pad :: Maybe Int)]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 6)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_LibSVMIter" dummy =
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
            forall a . Fullfilled "_LibSVMIter" () a =>
              ArgsHMap "_LibSVMIter" () a -> IO DataIterHandle
_LibSVMIter args
  = let allargs
          = catMaybes
              [("data_libsvm",) . showValue <$>
                 (args !? #data_libsvm :: Maybe Text),
               ("data_shape",) . showValue <$>
                 (args !? #data_shape :: Maybe [Int]),
               ("label_libsvm",) . showValue <$>
                 (args !? #label_libsvm :: Maybe Text),
               ("label_shape",) . showValue <$>
                 (args !? #label_shape :: Maybe [Int]),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("round_batch",) . showValue <$>
                 (args !? #round_batch :: Maybe Bool),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"])))]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 7)
         mxDataIterCreateIter di keys vals

type instance ParameterList "_MNISTIter" dummy =
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
           forall a . Fullfilled "_MNISTIter" () a =>
             ArgsHMap "_MNISTIter" () a -> IO DataIterHandle
_MNISTIter args
  = let allargs
          = catMaybes
              [("image",) . showValue <$> (args !? #image :: Maybe Text),
               ("label",) . showValue <$> (args !? #label :: Maybe Text),
               ("batch_size",) . showValue <$> (args !? #batch_size :: Maybe Int),
               ("shuffle",) . showValue <$> (args !? #shuffle :: Maybe Bool),
               ("flat",) . showValue <$> (args !? #flat :: Maybe Bool),
               ("seed",) . showValue <$> (args !? #seed :: Maybe Int),
               ("silent",) . showValue <$> (args !? #silent :: Maybe Bool),
               ("num_parts",) . showValue <$> (args !? #num_parts :: Maybe Int),
               ("part_index",) . showValue <$> (args !? #part_index :: Maybe Int),
               ("prefetch_buffer",) . showValue <$>
                 (args !? #prefetch_buffer :: Maybe Int),
               ("ctx",) . showValue <$>
                 (args !? #ctx :: Maybe (EnumType '["cpu", "gpu"])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                              "int8", "uint8"])))]
        (keys, vals) = unzip allargs
      in
      do dis <- mxListDataIters
         di <- return (dis !! 8)
         mxDataIterCreateIter di keys vals