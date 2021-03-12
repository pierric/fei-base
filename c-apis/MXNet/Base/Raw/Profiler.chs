module MXNet.Base.Raw.Profiler where

import RIO
import RIO.List (unzip)
import Foreign.Marshal (alloca)
import Foreign.C.Types (CInt)
import Foreign.C.String (CString)
import Foreign.Storable (Storable(..))

{# import MXNet.Base.Raw.Common #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{#
fun MXSetProfilerConfig as mxSetProfilerConfig_
    {
        `CInt',
        withCStringArrayT* `[Text]',
        withCStringArrayT* `[Text]'
    } -> `CInt'
#}

mxSetProfilerConfig :: HasCallStack => [(Text, Text)] -> IO ()
mxSetProfilerConfig kwargs = do
    let num = fromIntegral $ length kwargs
        (keys, vals) = unzip kwargs
    checked $ mxSetProfilerConfig_ num keys vals

{#
fun MXSetProfilerState as mxSetProfilerState_
    {
        `CInt'
    } -> `CInt'
#}

mxSetProfilerState :: HasCallStack => Int -> IO ()
mxSetProfilerState = checked . mxSetProfilerState_ . fromIntegral

{#
fun MXDumpProfile as mxDumpProfile_
    {
        `CInt'
    } -> `CInt'
#}

mxDumpProfile :: HasCallStack => Int -> IO ()
mxDumpProfile = checked . mxDumpProfile_ . fromIntegral

{#
fun MXAggregateProfileStatsPrint as mxAggregateProfileStatsPrint_
    {
        alloca- `Text' peekCStringPtrT*,
        `CInt'
    } -> `CInt'
#}

mxAggregateProfileStatsPrint :: HasCallStack => Int -> IO Text
mxAggregateProfileStatsPrint = fmap unWrapText
                                    . checked
                                    . fmap (second WrapText)
                                    . mxAggregateProfileStatsPrint_
                                    . fromIntegral

{#
fun MXAggregateProfileStatsPrintEx as mxAggregateProfileStatsPrintEx_
    {
        alloca- `C2HSImp.Ptr CString' id,
        `CInt',
        `CInt',
        `CInt',
        `CInt'
    } -> `CInt'
#}

mxAggregateProfileStatsPrintEx :: HasCallStack => Int -> Int -> Int -> Int -> IO Text
mxAggregateProfileStatsPrintEx reset format sort_by ascending = do
    let reset'     = fromIntegral reset
        format'    = fromIntegral format
        sort_by'   = fromIntegral sort_by
        ascending' = fromIntegral ascending
    (rc, pcstr) <- mxAggregateProfileStatsPrintEx_ reset' format' sort_by' ascending'
    checkRC rc
    cstr <- peek pcstr
    peekCStringT cstr

{#
fun MXProfilePause as mxProfilePause_
    {
        `CInt'
    } -> `CInt'
#}

mxProfilePause :: HasCallStack => Int -> IO ()
mxProfilePause = checked . mxProfilePause_ . fromIntegral
