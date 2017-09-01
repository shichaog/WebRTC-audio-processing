lib_LTLIBRARIES = libwebrtc_audio_processing.la

libwebrtc_audio_processing_la_SOURCES = include/audio_processing.h \
					aec/include/echo_cancellation.h \
					aec/aec_common.h \
					aec/aec_core.c \
					aec/aec_core.h \
					aec/aec_core_internal.h \
					aec/aec_rdft.c \
					aec/aec_rdft.h \
					aec/aec_resampler.c \
					aec/aec_resampler.h \
					aec/echo_cancellation.c \
					aec/echo_cancellation_internal.h \
					aecm/include/echo_control_mobile.h \
					aecm/echo_control_mobile.c \
					aecm/aecm_core.c \
					aecm/aecm_core.h \
					aecm/aecm_core_c.c \
					aecm/aecm_defines.h \
					agc/legacy/analog_agc.c \
					agc/legacy/analog_agc.h \
					agc/legacy/gain_control.h \
					agc/legacy/digital_agc.c \
					agc/legacy/digital_agc.h \
					agc/agc.cc \
					agc/agc.h \
					agc/agc_manager_direct.cc \
					agc/agc_manager_direct.h \
					agc/gain_map_internal.h \
					agc/histogram.cc \
					agc/histogram.h \
					agc/utility.cc \
					agc/utility.h \
					beamformer/array_util.h \
					beamformer/beamformer.h \
					beamformer/complex_matrix.h \
					beamformer/covariance_matrix_generator.h \
					beamformer/matrix.h \
					beamformer/matrix_test_helpers.h \
					beamformer/nonlinear_beamformer.h \
					beamformer/array_util.cc \
					beamformer/covariance_matrix_generator.cc \
					beamformer/nonlinear_beamformer.cc \
					intelligibility/intelligibility_enhancer.h \
					intelligibility/intelligibility_utils.h \
					intelligibility/intelligibility_enhancer.cc \
					intelligibility/intelligibility_utils.cc \
					logging/aec_logging.h \
					logging/aec_logging_file_handling.h \
					logging/aec_logging_file_handling.cc \
					transient/common.h \
					transient/daubechies_8_wavelet_coeffs.h \
					transient/dyadic_decimator.h \
					transient/file_utils.h \
					transient/moving_moments.h \
					transient/transient_detector.h \
					transient/transient_suppressor.h \
					transient/wpd_node.h \
					transient/wpd_tree.h \
					transient/click_annotate.cc \
					transient/file_utils.cc \
					transient/moving_moments.cc \
					transient/transient_detector.cc \
					transient/transient_suppressor.cc \
					transient/wpd_node.cc \
					transient/wpd_tree.cc \
					utility/delay_estimator.c \
					utility/delay_estimator.h \
					utility/delay_estimator_internal.h \
					utility/delay_estimator_wrapper.c \
					utility/delay_estimator_wrapper.h \
					vad/common.h \
					vad/gmm.h \
					vad/noise_gmm_tables.h \
					vad/pitch_based_vad.h \
					vad/pitch_internal.h \
					vad/pole_zero_filter.h \
					vad/standalone_vad.h \
					vad/vad_audio_proc.h \
					vad/vad_audio_proc_internal.h \
					vad/vad_circular_buffer.h \
					vad/voice_activity_detector.h \
					vad/voice_gmm_tables.h \
					vad/gmm.cc \
					vad/pitch_based_vad.cc \
					vad/pitch_internal.cc \
					vad/pole_zero_filter.cc \
					vad/standalone_vad.cc \
					vad/vad_audio_proc.cc \
					vad/vad_circular_buffer.cc \
					vad/voice_activity_detector.cc \
					audio_buffer.cc \
					audio_buffer.h \
					audio_processing_impl.cc \
					audio_processing_impl.h \
					common.h \
					echo_cancellation_impl.cc \
					echo_cancellation_impl.h \
					echo_control_mobile_impl.cc \
					echo_control_mobile_impl.h \
					gain_control_impl.cc \
					gain_control_impl.h \
					high_pass_filter_impl.cc \
					high_pass_filter_impl.h \
					level_estimator_impl.cc \
					level_estimator_impl.h \
					noise_suppression_impl.cc \
					noise_suppression_impl.h \
					rms_level.cc \
					rms_level.h \
					splitting_filter.cc \
					splitting_filter.h \
					processing_component.cc \
					processing_component.h \
					three_band_filter_bank.cc \
					three_band_filter_bank.h \
					typing_detection.cc \
					typing_detection.h \
					voice_detection_impl.cc \
					voice_detection_impl.h

if NS_FIXED
COMMON_CFLAGS += -DWEBRTC_NS_FIXED=1
COMMON_CXXFLAGS += -DWEBRTC_NS_FIXED=1
libwebrtc_audio_processing_la_SOURCES += \
					ns/include/noise_suppression_x.h \
					ns/noise_suppression_x.c \
					ns/nsx_defines.h \
					ns/nsx_core.c \
					ns/nsx_core.h \
					ns/nsx_core_c.c
if HAVE_NEON
libwebrtc_audio_processing_la_SOURCES += \
					ns/nsx_core_neon.c
endif
else
COMMON_CFLAGS += -DWEBRTC_NS_FLOAT=1
COMMON_CXXFLAGS += -DWEBRTC_NS_FLOAT=1
libwebrtc_audio_processing_la_SOURCES += \
					ns/include/noise_suppression.h \
					ns/noise_suppression.c \
					ns/defines.h \
					ns/ns_core.c \
					ns/ns_core.h \
					ns/windows_private.h
endif

if HAVE_X86
noinst_LTLIBRARIES = libwebrtc_audio_processing_privatearch.la
LIBWEBRTC_PRIVATEARCH=libwebrtc_audio_processing_privatearch.la
libwebrtc_audio_processing_privatearch_la_SOURCES = \
					aec/aec_core_sse2.c \
					aec/aec_rdft_sse2.c
libwebrtc_audio_processing_privatearch_la_CFLAGS = $(AM_CFLAGS) $(COMMON_CFLAGS) -msse2
libwebrtc_audio_processing_privatearch_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS) -msse2
libwebrtc_audio_processing_privatearch_la_LDFLAGS = $(AM_LDFLAGS)
endif

if HAVE_NEON
libwebrtc_audio_processing_la_SOURCES += \
					aec/aec_core_neon.c \
					aec/aec_rdft_neon.c \
					aecm/aecm_core_neon.c
endif

libwebrtc_audio_processing_la_CFLAGS = $(AM_CFLAGS) $(COMMON_CFLAGS)
libwebrtc_audio_processing_la_CXXFLAGS = $(AM_CXXFLAGS) $(COMMON_CXXFLAGS)

libwebrtc_audio_processing_la_LIBADD = $(top_builddir)/webrtc/libwebrtc.la \
				       $(top_builddir)/webrtc/base/libbase.la \
				       $(top_builddir)/webrtc/system_wrappers/libsystem_wrappers.la \
				       $(top_builddir)/webrtc/common_audio/libcommon_audio.la \
				       $(top_builddir)/webrtc/modules/audio_coding/libaudio_coding.la \
				       $(LIBWEBRTC_PRIVATEARCH) \
				       $(GNUSTL_LIBS)
libwebrtc_audio_processing_la_LDFLAGS = $(AM_LDFLAGS) $(COMMON_LDFLAGS) \
					-no-undefined \
					-version-info $(LIBWEBRTC_AUDIO_PROCESSING_VERSION_INFO)

# FIXME: The MIPS optimisations need to be hooked up once we have the
# autotools conditionals in place
EXTRA_DIST = BUILD.gn \
	     aec/aec_core_mips.c \
	     aec/aec_rdft_neon.c \
	     aecm/aecm_core_mips.c \
	     ns/nsx_core_mips.c
