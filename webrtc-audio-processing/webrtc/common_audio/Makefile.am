noinst_LTLIBRARIES = libcommon_audio.la

noinst_HEADERS = include/audio_util.h

libcommon_audio_la_SOURCES = resampler/include/push_resampler.h \
			     resampler/include/resampler.h \
			     resampler/push_sinc_resampler.h \
			     resampler/sinc_resampler.h \
			     resampler/sinusoidal_linear_chirp_source.h \
			     resampler/push_resampler.cc \
			     resampler/push_sinc_resampler.cc \
			     resampler/resampler.cc \
			     resampler/sinc_resampler.cc \
			     resampler/sinusoidal_linear_chirp_source.cc \
			     signal_processing/include/real_fft.h \
			     signal_processing/include/signal_processing_library.h \
			     signal_processing/include/spl_inl.h \
			     signal_processing/include/spl_inl_armv7.h \
			     signal_processing/include/spl_inl_mips.h \
			     signal_processing/auto_corr_to_refl_coef.c \
			     signal_processing/auto_correlation.c \
			     signal_processing/complex_fft.c \
			     signal_processing/complex_fft_tables.h \
			     signal_processing/copy_set_operations.c \
			     signal_processing/cross_correlation.c \
			     signal_processing/division_operations.c \
			     signal_processing/dot_product_with_scale.c \
			     signal_processing/downsample_fast.c \
			     signal_processing/energy.c \
			     signal_processing/filter_ar.c \
			     signal_processing/filter_ma_fast_q12.c \
			     signal_processing/get_hanning_window.c \
			     signal_processing/get_scaling_square.c \
			     signal_processing/ilbc_specific_functions.c \
			     signal_processing/levinson_durbin.c \
			     signal_processing/lpc_to_refl_coef.c \
			     signal_processing/min_max_operations.c \
			     signal_processing/randomization_functions.c \
			     signal_processing/real_fft.c \
			     signal_processing/refl_coef_to_lpc.c \
			     signal_processing/resample.c \
			     signal_processing/resample_48khz.c \
			     signal_processing/resample_by_2.c \
			     signal_processing/resample_by_2_internal.c \
			     signal_processing/resample_by_2_internal.h \
			     signal_processing/resample_fractional.c \
			     signal_processing/spl_init.c \
			     signal_processing/spl_sqrt.c \
			     signal_processing/splitting_filter.c \
			     signal_processing/sqrt_of_one_minus_x_squared.c \
			     signal_processing/vector_scaling_operations.c \
			     vad/include/vad.h \
			     vad/include/webrtc_vad.h \
			     vad/vad.cc \
			     vad/vad_core.c \
			     vad/vad_core.h \
			     vad/vad_filterbank.c \
			     vad/vad_filterbank.h \
			     vad/vad_gmm.c \
			     vad/vad_gmm.h \
			     vad/vad_sp.c \
			     vad/vad_sp.h \
			     vad/webrtc_vad.c \
			     audio_converter.cc \
			     audio_converter.h \
			     audio_ring_buffer.cc \
			     audio_ring_buffer.h \
			     audio_util.cc \
			     blocker.cc \
			     blocker.h \
			     channel_buffer.cc \
			     channel_buffer.h \
			     fft4g.c \
			     fft4g.h \
			     fir_filter.cc \
			     fir_filter.h \
			     lapped_transform.cc \
			     lapped_transform.h \
			     real_fourier.cc \
			     real_fourier.h \
			     real_fourier_ooura.cc \
			     real_fourier_ooura.h \
			     real_fourier_openmax.h \
			     ring_buffer.h \
			     ring_buffer.c \
			     sparse_fir_filter.cc \
			     sparse_fir_filter.h \
			     wav_file.h \
			     wav_file.cc \
			     wav_header.h \
			     wav_header.cc \
			     window_generator.h \
			     window_generator.cc

if HAVE_X86
noinst_LTLIBRARIES += libcommon_audio_sse2.la
libcommon_audio_sse2_la_SOURCES = \
			     resampler/sinc_resampler_sse.cc \
			     fir_filter_sse.cc \
			     fir_filter_sse.h

libcommon_audio_sse2_la_CFLAGS = $(AM_CFLAGS) $(COMMON_CFLAGS) -msse2
libcommon_audio_sse2_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS) -msse2
libcommon_audio_sse2_la_LDFLAGS = $(AM_LDFLAGS)

libcommon_audio_la_LIBADD = libcommon_audio_sse2.la
endif

if HAVE_ARM
libcommon_audio_la_SOURCES += \
			     signal_processing/complex_bit_reverse_arm.S \
			     signal_processing/spl_sqrt_floor_arm.S
endif

if HAVE_ARMV7
libcommon_audio_la_SOURCES += \
			     signal_processing/filter_ar_fast_q12_armv7.S
endif

if HAVE_NEON
libcommon_audio_la_SOURCES += \
			     resampler/sinc_resampler_neon.cc \
			     signal_processing/cross_correlation_neon.c \
			     signal_processing/downsample_fast_neon.c \
			     signal_processing/min_max_operations_neon.c \
			     fir_filter_neon.cc \
			     fir_filter_neon.h
endif

if !HAVE_ARM
# FIXME: This condition will also hold for !HAVE_MIPS
libcommon_audio_la_SOURCES += \
			     signal_processing/complex_bit_reverse.c \
			     signal_processing/filter_ar_fast_q12.c \
			     signal_processing/spl_sqrt_floor.c
endif

libcommon_audio_la_CCASFLAGS = $(AM_CCASFLAGS) $(COMMON_CFLAGS)
libcommon_audio_la_CFLAGS = $(AM_CFLAGS) $(COMMON_CFLAGS)
libcommon_audio_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS)

# FIXME: The MIPS optimisations need to be hooked up once we have the
# autotools conditionals in place
EXTRA_DIST = BUILD.gn \
	     signal_processing/complex_bit_reverse_mips.c \
	     signal_processing/complex_fft_mips.c \
	     signal_processing/cross_correlation_mips.c \
	     signal_processing/downsample_fast_mips.c \
	     signal_processing/filter_ar_fast_q12_mips.c \
	     signal_processing/min_max_operations_mips.c \
	     signal_processing/resample_by_2_mips.c \
	     signal_processing/spl_sqrt_floor_mips.c \
	     signal_processing/vector_scaling_operations_mips.c
