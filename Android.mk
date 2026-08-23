LOCAL_PATH:= $(call my-dir)

SJPEG_CFLAGS := -Wall -DANDROID -DHAVE_MALLOC_H -DHAVE_PTHREAD
SJPEG_CFLAGS += -fvisibility=hidden

ifeq ($(APP_OPTIM),release)
  SJPEG_CFLAGS += -finline-functions -ffast-math \
                  -ffunction-sections -fdata-sections
  ifeq ($(findstring clang,$(NDK_TOOLCHAIN_VERSION)),)
    SJPEG_CFLAGS += -frename-registers -s
  endif
endif

ifneq ($(findstring armeabi-v7a, $(TARGET_ARCH_ABI)),)
  # Setting LOCAL_ARM_NEON will enable -mfpu=neon which may cause illegal
  # instructions to be generated for armv7a code. Instead target the neon code
  # specifically.
  NEON := cc.neon
  USE_CPUFEATURES := yes
else
  NEON := cc
endif

enc_srcs := \
        src/api.cc \
        src/bit_writer.cc \
        src/colors_rgb.$(NEON) \
        src/enc.cc \
        src/encoders.cc \
        src/entropy.cc \
        src/fdct.$(NEON) \
        src/headers.cc \
        src/histogram.$(NEON) \
        src/dichotomy.cc \
        src/jpeg_tools.cc \
        src/quantize.$(NEON) \
        src/yuv_convert.$(NEON) \
        src/score_7.cc \

# AVX2 kernel variants (x86 only, opt-in: 'ndk-build SJPEG_HAVE_AVX2=1').
# A separate module, not part of enc_srcs above, because AVX2 needs its own
# -mavx2 that must NOT apply to the rest of the library (unlike SSE2, AVX2
# isn't a safe baseline assumption even on x86_64 -- support is picked at
# runtime by SupportsAVX2() in src/enc.cc, same as the desktop build).
ifneq ($(filter x86 x86_64, $(TARGET_ARCH_ABI)),)
  ifeq ($(SJPEG_HAVE_AVX2),1)
    include $(CLEAR_VARS)
    LOCAL_SRC_FILES := src/histogram_avx2.cc src/quantize_avx2.cc \
                       src/riskiness_avx2.cc
    LOCAL_CFLAGS := $(SJPEG_CFLAGS) -mavx2
    LOCAL_C_INCLUDES += $(LOCAL_PATH)/src
    LOCAL_MODULE := sjpeg_avx2
    include $(BUILD_STATIC_LIBRARY)
    SJPEG_CFLAGS += -DSJPEG_HAVE_AVX2
    SJPEG_AVX2_LIB := sjpeg_avx2
  endif
endif

################################################################################

include $(CLEAR_VARS)

LOCAL_SRC_FILES := \
    $(enc_srcs) \

LOCAL_CFLAGS := $(SJPEG_CFLAGS)
LOCAL_C_INCLUDES += $(LOCAL_PATH)/src
LOCAL_WHOLE_STATIC_LIBRARIES := $(SJPEG_AVX2_LIB)

# prefer arm over thumb mode for performance gains
LOCAL_ARM_MODE := arm

LOCAL_MODULE := sjpeg_static

include $(BUILD_STATIC_LIBRARY)

ifeq ($(ENABLE_SHARED),1)
  include $(CLEAR_VARS)
  LOCAL_WHOLE_STATIC_LIBRARIES := sjpeg_static
  LOCAL_MODULE := sjpeg

  include $(BUILD_SHARED_LIBRARY)
endif

################################################################################

include $(LOCAL_PATH)/examples/Android.mk

$(call import-module,android/cpufeatures)
