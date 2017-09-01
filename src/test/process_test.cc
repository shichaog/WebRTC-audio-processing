/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>

#include <sys/stat.h>

#include <algorithm>

#include "webrtc/base/scoped_ptr.h"
#include "test_utils.h"
#include "webrtc/base/wav_file.h"
#include "webrtc/base/audio_processing_impl.h"
#include "webrtc/common.h"
#include "webrtc/modules/audio_processing/include/audio_processing.h"
#include "webrtc/modules/audio_processing/test/test_utils.h"
#include "webrtc/modules/interface/module_common_types.h"
#include "webrtc/test/testsupport/fileutils.h"
#include "webrtc/test/testsupport/perf_test.h"

#define EXPECT_OP(op, val1, val2)                                       \
	do {                                                                  \
		if (!((val1) op (val2))) {                                          \
			fprintf(stderr, "Check failed: %s %s %s\n", #val1, #op, #val2);   \
			exit(1);                                                          \
		}                                                                   \
	} while (0)

#define EXPECT_EQ(val1, val2)  EXPECT_OP(==, val1, val2)
#define EXPECT_NE(val1, val2)  EXPECT_OP(!=, val1, val2)
#define EXPECT_GT(val1, val2)  EXPECT_OP(>, val1, val2)
#define EXPECT_LT(val1, val2)  EXPECT_OP(<, val1, val2)

void usage() {
  printf(
  "Usage: process_test [options] [-pb PROTOBUF_FILE]\n"
  "  [-ir REVERSE_FILE] [-i PRIMARY_FILE] [-o OUT_FILE]\n");
  printf(
  "process_test is a test application for AudioProcessing.\n\n"
  "When a protobuf debug file is available, specify it with -pb. Alternately,\n"
  "when -ir or -i is used, the specified files will be processed directly in\n"
  "a simulation mode. Otherwise the full set of legacy test files is expected\n"
  "to be present in the working directory. OUT_FILE should be specified\n"
  "without extension to support both raw and wav output.\n\n");
  printf("Options\n");
  printf("General configuration (only used for the simulation mode):\n");
  printf("  -fs SAMPLE_RATE_HZ\n");
  printf("  -ch CHANNELS_IN CHANNELS_OUT\n");
  printf("  -rch REVERSE_CHANNELS\n");
  printf("\n");
  printf("Component configuration:\n");
  printf(
  "All components are disabled by default. Each block below begins with a\n"
  "flag to enable the component with default settings. The subsequent flags\n"
  "in the block are used to provide configuration settings.\n");
  printf("\n  -aec     Echo cancellation\n");
  printf("  --drift_compensation\n");
  printf("  --no_drift_compensation\n");
  printf("  --no_echo_metrics\n");
  printf("  --no_delay_logging\n");
  printf("  --aec_suppression_level LEVEL  [0 - 2]\n");
  printf("  --extended_filter\n");
  printf("  --no_reported_delay\n");
  printf("\n  -aecm    Echo control mobile\n");
  printf("  --aecm_echo_path_in_file FILE\n");
  printf("  --aecm_echo_path_out_file FILE\n");
  printf("  --no_comfort_noise\n");
  printf("  --routing_mode MODE  [0 - 4]\n");
  printf("\n  -agc     Gain control\n");
  printf("  --analog\n");
  printf("  --adaptive_digital\n");
  printf("  --fixed_digital\n");
  printf("  --target_level LEVEL\n");
  printf("  --compression_gain GAIN\n");
  printf("  --limiter\n");
  printf("  --no_limiter\n");
  printf("\n  -hpf     High pass filter\n");
  printf("\n  -ns      Noise suppression\n");
  printf("  --ns_low\n");
  printf("  --ns_moderate\n");
  printf("  --ns_high\n");
  printf("  --ns_very_high\n");
  printf("  --ns_prob_file FILE\n");
  printf("\n  -vad     Voice activity detection\n");
  printf("  --vad_out_file FILE\n");
  printf("\n  -expns   Experimental noise suppression\n");
  printf("\n Level metrics (enabled by default)\n");
  printf("  --no_level_metrics\n");
  printf("\n");
  printf("Modifiers:\n");
  printf("  --noasm            Disable SSE optimization.\n");
  printf("  --add_delay DELAY  Add DELAY ms to input value.\n");
  printf("  --delay DELAY      Override input delay with DELAY ms.\n");
  printf("  --perf             Measure performance.\n");
  printf("  --quiet            Suppress text output.\n");
  printf("  --no_progress      Suppress progress.\n");
  printf("  --raw_output       Raw output instead of WAV file.\n");
  printf("  --debug_file FILE  Dump a debug recording.\n");
}

static float MicLevel2Gain(int level) {
  return pow(10.0f, ((level - 127.0f) / 128.0f * 40.0f) / 20.0f);
}

static void SimulateMic(int mic_level, webrtc::AudioFrame* frame) {
  mic_level = std::min(std::max(mic_level, 0), 255);
  float mic_gain = MicLevel2Gain(mic_level);
  int num_samples = frame->samples_per_channel_ * frame->num_channels_;
  float v;
  for (int n = 0; n < num_samples; n++) {
    v = floor(frame->data_[n] * mic_gain + 0.5);
    v = std::max(std::min(32767.0f, v), -32768.0f);
    frame->data_[n] = static_cast<int16_t>(v);
  }
}

// void function for gtest.
int main(int argc, char* argv[]) {
  int16_t event = 0;
  if (argc > 1 && strcmp(argv[1], "--help") == 0) {
    usage();
    return 0;
  }

  if (argc < 2) {
    printf("Try `process_test --help' for more information.\n\n");
  }

  rtc::scoped_ptr<webrtc::AudioProcessing> apm(webrtc::AudioProcessing::Create());

  const char* far_filename = NULL;
  const char* near_filename = NULL;
  std::string out_filename;
  const char* vad_out_filename = NULL;
  const char* ns_prob_filename = NULL;
  const char* aecm_echo_path_in_filename = NULL;
  const char* aecm_echo_path_out_filename = NULL;

  int32_t sample_rate_hz = 16000;

  int num_capture_input_channels = 1;
  int num_capture_output_channels = 1;
  int num_render_channels = 1;

  int samples_per_channel = sample_rate_hz / 100;

  bool simulating = false;
  bool perf_testing = false;
  bool verbose = true;
  bool progress = true;
  bool raw_output = false;
  int extra_delay_ms = 0;
  int override_delay_ms = 0;
  webrtc::Config config;

  apm->level_estimator()->Enable(true);
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ir") == 0) {
      i++;
      far_filename = argv[i];
      simulating = true;
    } else if (strcmp(argv[i], "-i") == 0) {
      i++;
      near_filename = argv[i];
      simulating = true;
    } else if (strcmp(argv[i], "-o") == 0) {
      i++;
      out_filename = argv[i];
    } else if (strcmp(argv[i], "-fs") == 0) {
      i++;
      sscanf(argv[i], "%d", &sample_rate_hz);
      samples_per_channel = sample_rate_hz / 100;
    } else if (strcmp(argv[i], "-ch") == 0) {
      i++;
      sscanf(argv[i], "%d", &num_capture_input_channels);
      i++;
      sscanf(argv[i], "%d", &num_capture_output_channels);
    } else if (strcmp(argv[i], "-rch") == 0) {
      i++;
      sscanf(argv[i], "%d", &num_render_channels);
    } else if (strcmp(argv[i], "-aec") == 0) {
      apm->echo_cancellation()->Enable(true);
      apm->echo_cancellation()->enable_metrics(true);
      apm->echo_cancellation()->enable_delay_logging(true);
    } else if (strcmp(argv[i], "--drift_compensation") == 0) {
      apm->echo_cancellation()->Enable(true);
      // TODO(ajm): this is enabled in the VQE test app by default. Investigate
      //            why it can give better performance despite passing zeros.
      apm->echo_cancellation()->enable_drift_compensation(true);
    } else if (strcmp(argv[i], "--no_drift_compensation") == 0) {
      apm->echo_cancellation()->Enable(true);
      apm->echo_cancellation()->enable_drift_compensation(false);
    } else if (strcmp(argv[i], "--no_echo_metrics") == 0) {
      apm->echo_cancellation()->Enable(true);
      apm->echo_cancellation()->enable_metrics(false);
    } else if (strcmp(argv[i], "--no_delay_logging") == 0) {
      apm->echo_cancellation()->Enable(true);
      apm->echo_cancellation()->enable_delay_logging(false);
    } else if (strcmp(argv[i], "--no_level_metrics") == 0) {
      apm->level_estimator()->Enable(false);
    } else if (strcmp(argv[i], "--aec_suppression_level") == 0) {
      i++;
      int suppression_level;
      sscanf(argv[i], "%d", &suppression_level);
      apm->echo_cancellation()->set_suppression_level(
      static_cast<webrtc::EchoCancellation::SuppressionLevel>(
                        suppression_level));
    } else if (strcmp(argv[i], "--extended_filter") == 0) {
      config.Set<webrtc::ExtendedFilter>(new webrtc::ExtendedFilter(true));
    } else if (strcmp(argv[i], "--no_reported_delay") == 0) {
      config.Set<webrtc::DelayAgnostic>(new webrtc::DelayAgnostic(true));
    } else if (strcmp(argv[i], "--delay_agnostic") == 0) {
      config.Set<webrtc::DelayAgnostic>(new webrtc::DelayAgnostic(true));
    } else if (strcmp(argv[i], "-aecm") == 0) {
      apm->echo_control_mobile()->Enable(true);
    } else if (strcmp(argv[i], "--aecm_echo_path_in_file") == 0) {
      i++;
      aecm_echo_path_in_filename = argv[i];
    } else if (strcmp(argv[i], "--aecm_echo_path_out_file") == 0) {
      i++;
      aecm_echo_path_out_filename = argv[i];
    } else if (strcmp(argv[i], "--no_comfort_noise") == 0) {
                apm->echo_control_mobile()->enable_comfort_noise(false);
    } else if (strcmp(argv[i], "--routing_mode") == 0) {
      i++;
      int routing_mode;
      sscanf(argv[i], "%d", &routing_mode);
                apm->echo_control_mobile()->set_routing_mode(
                    static_cast<webrtc::EchoControlMobile::RoutingMode>(
                        routing_mode));

    } else if (strcmp(argv[i], "-agc") == 0) {
      apm->gain_control()->Enable(true);

    } else if (strcmp(argv[i], "--analog") == 0) {
      apm->gain_control()->Enable(true);
                apm->gain_control()->set_mode(webrtc::GainControl::kAdaptiveAnalog);

    } else if (strcmp(argv[i], "--adaptive_digital") == 0) {
      apm->gain_control()->Enable(true);
                apm->gain_control()->set_mode(webrtc::GainControl::kAdaptiveDigital);

    } else if (strcmp(argv[i], "--fixed_digital") == 0) {
      apm->gain_control()->Enable(true);
                apm->gain_control()->set_mode(webrtc::GainControl::kFixedDigital);

    } else if (strcmp(argv[i], "--target_level") == 0) {
      i++;
      int level;
      sscanf(argv[i], "%d", &level);

      apm->gain_control()->Enable(true);
                apm->gain_control()->set_target_level_dbfs(level);

    } else if (strcmp(argv[i], "--compression_gain") == 0) {
      i++;
      int gain;
      sscanf(argv[i], "%d", &gain);

      apm->gain_control()->Enable(true);
                apm->gain_control()->set_compression_gain_db(gain);

    } else if (strcmp(argv[i], "--limiter") == 0) {
      apm->gain_control()->Enable(true);
                apm->gain_control()->enable_limiter(true);

    } else if (strcmp(argv[i], "--no_limiter") == 0) {
      apm->gain_control()->Enable(true);
                apm->gain_control()->enable_limiter(false);

    } else if (strcmp(argv[i], "-hpf") == 0) {
      apm->high_pass_filter()->Enable(true);

    } else if (strcmp(argv[i], "-ns") == 0) {
      apm->noise_suppression()->Enable(true);

    } else if (strcmp(argv[i], "--ns_low") == 0) {
      apm->noise_suppression()->Enable(true);
          apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kLow);

    } else if (strcmp(argv[i], "--ns_moderate") == 0) {
      apm->noise_suppression()->Enable(true);
          apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kModerate);

    } else if (strcmp(argv[i], "--ns_high") == 0) {
      apm->noise_suppression()->Enable(true);
          apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kHigh);

    } else if (strcmp(argv[i], "--ns_very_high") == 0) {
     apm->kNoError, apm->noise_suppression()->Enable(true);
          apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kVeryHigh);

    } else if (strcmp(argv[i], "--ns_prob_file") == 0) {
      i++;
      ns_prob_filename = argv[i];

    } else if (strcmp(argv[i], "-vad") == 0) {
      apm->voice_detection()->Enable(true);

    } else if (strcmp(argv[i], "--vad_very_low") == 0) {
      apm->voice_detection()->Enable(true);
          apm->voice_detection()->set_likelihood(webrtc::VoiceDetection::kVeryLowLikelihood);

    } else if (strcmp(argv[i], "--vad_low") == 0) {
      apm->voice_detection()->Enable(true);
          apm->voice_detection()->set_likelihood(webrtc::VoiceDetection::kLowLikelihood);

    } else if (strcmp(argv[i], "--vad_moderate") == 0) {
      apm->voice_detection()->Enable(true);
          apm->voice_detection()->set_likelihood(webrtc::VoiceDetection::kModerateLikelihood);

    } else if (strcmp(argv[i], "--vad_high") == 0) {
      apm->voice_detection()->Enable(true);
          apm->voice_detection()->set_likelihood(webrtc::VoiceDetection::kHighLikelihood);

    } else if (strcmp(argv[i], "--vad_out_file") == 0) {
      i++;
      vad_out_filename = argv[i];

    } else if (strcmp(argv[i], "-expns") == 0) {
      config.Set<webrtc::ExperimentalNs>(new webrtc::ExperimentalNs(true));

    } else if (strcmp(argv[i], "--noasm") == 0) {
      // We need to reinitialize here if components have already been enabled.
      apm->Initialize();

    } else if (strcmp(argv[i], "--add_delay") == 0) {
      i++;
      sscanf(argv[i], "%d", &extra_delay_ms);

    } else if (strcmp(argv[i], "--delay") == 0) {
      i++;
      sscanf(argv[i], "%d", &override_delay_ms);

    } else if (strcmp(argv[i], "--perf") == 0) {
      perf_testing = true;

    } else if (strcmp(argv[i], "--quiet") == 0) {
      verbose = false;
      progress = false;

    } else if (strcmp(argv[i], "--no_progress") == 0) {
      progress = false;

    } else if (strcmp(argv[i], "--raw_output") == 0) {
      raw_output = true;

    } else if (strcmp(argv[i], "--debug_file") == 0) {
      i++;
      apm->StartDebugRecording(argv[i]);
    } else {
		;
    }
  }

  apm->SetExtraOptions(config);

  // If we're reading a protobuf file, ensure a simulation hasn't also
  // been requested (which makes no sense...)
  if (verbose) {
    printf("Sample rate: %d Hz\n", sample_rate_hz);
    printf("Primary channels: %d (in), %d (out)\n",
           num_capture_input_channels,
           num_capture_output_channels);
    printf("Reverse channels: %d \n", num_render_channels);
  }

  const std::string out_path = "proc_out.pcm";
  const char far_file_default[] = "apm_far.pcm";
  const char near_file_default[] = "apm_near.pcm";
  const char event_filename[] = "apm_event.dat";
  const char delay_filename[] = "apm_delay.dat";
  const char drift_filename[] = "apm_drift.dat";
  const std::string vad_file_default = out_path + "vad_out.dat";
  const std::string ns_prob_file_default = out_path + "ns_prob.dat";

  if (out_filename.size() == 0) {
    out_filename = out_path + "out";
  }

  if (!vad_out_filename) {
    vad_out_filename = vad_file_default.c_str();
  }

  if (!ns_prob_filename) {
    ns_prob_filename = ns_prob_file_default.c_str();
  }

  FILE* far_file = NULL;
  FILE* near_file = NULL;
  FILE* vad_out_file = NULL;

  rtc::scoped_ptr<webrtc::WavWriter> output_wav_file;
  rtc::scoped_ptr<webrtc::RawFile> output_raw_file;

  if (far_filename) {
      far_file = webrtc::OpenFile(far_filename, "rb");
  }

  near_file = webrtc::OpenFile(near_filename, "rb");

  int near_size_bytes = 0;
  struct stat st;
  stat(near_filename, &st);
  near_size_bytes = st.st_size;
/*
  if (apm->voice_detection()->is_enabled()) {
  	std::cout << "!!!!VAD detection enabled " << std::endl;	
    vad_out_file = webrtc::OpenFile(vad_out_filename, "wb");
  }

  if (apm->noise_suppression()->is_enabled()) {
  	std::cout << "!!!!NS enabled " << std::endl;	
    ns_prob_file = webrtc::OpenFile(ns_prob_filename, "wb");
  }

  if (aecm_echo_path_in_filename != NULL) {
  	std::cout << "!!!!aecm_echo_path_in_filename " << std::endl;	
    aecm_echo_path_in_file = webrtc::OpenFile(aecm_echo_path_in_filename, "rb");

    const size_t path_size =
        apm->echo_control_mobile()->echo_path_size_bytes();
    rtc::scoped_ptr<char[]> echo_path(new char[path_size]);
    fread(echo_path.get(), sizeof(char), path_size, aecm_echo_path_in_file);
              apm->echo_control_mobile()->SetEchoPath(echo_path.get(), path_size);
    fclose(aecm_echo_path_in_file);
    aecm_echo_path_in_file = NULL;
  }

  if (aecm_echo_path_out_filename != NULL) {
    aecm_echo_path_out_file = webrtc::OpenFile(aecm_echo_path_out_filename, "wb");
  }*/

  size_t read_count = 0;
  int reverse_count = 0;
  int primary_count = 0;
  int near_read_bytes = 0;

  webrtc::AudioFrame far_frame;
  webrtc::AudioFrame near_frame;

  int delay_ms = 0;
  int drift_samples = 0;
  int capture_level = 127;

  far_frame.sample_rate_hz_ = sample_rate_hz;
  far_frame.samples_per_channel_ = samples_per_channel;
  far_frame.num_channels_ = num_render_channels;
  near_frame.sample_rate_hz_ = sample_rate_hz;
  near_frame.samples_per_channel_ = samples_per_channel;
  near_frame.num_channels_ = num_capture_input_channels;
  size_t size = samples_per_channel * num_render_channels;

  while (simulating) {
      
      read_count = fread(far_frame.data_, sizeof(int16_t), size, far_file);

      if (read_count != size) {
        // Read an equal amount from the near file to avoid errors due to
        // not reaching end-of-file.
        fseek(near_file, read_count * sizeof(int16_t), SEEK_CUR);
            break;  // This is expected.
      }
      apm->ProcessReverseStream(&far_frame);
	  
      read_count = fread(near_frame.data_, sizeof(int16_t), size, near_file);
	  near_read_bytes += read_count * sizeof(int16_t);

      if (read_count != size) {
          break;  // This is expected.
      }

      apm->set_stream_delay_ms(delay_ms);
	  //std::cout << "!!!!!!!!!!drift_samples"<< drift_samples << std::endl;
      apm->echo_cancellation()->set_stream_drift_samples(drift_samples);

      int err = apm->ProcessStream(&near_frame);

      if (!raw_output && !output_wav_file) {
        output_wav_file.reset(new webrtc::WavWriter(out_filename + ".wav", sample_rate_hz, num_capture_output_channels));
      }
      webrtc::WriteIntData(near_frame.data_, size, output_wav_file.get(), output_raw_file.get());
    }
  printf("100%% complete\r");

 /* if (verbose) {
    printf("\nProcessed frames: %d (primary), %d (reverse)\n",
        primary_count, reverse_count);

    if (apm->level_estimator()->is_enabled()) {
      printf("\n--Level metrics--\n");
      printf("RMS: %d dBFS\n", -apm->level_estimator()->RMS());
    }
    if (apm->echo_cancellation()->is_delay_logging_enabled()) {
      int median = 0;
      int std = 0;
      float fraction_poor_delays = 0;
      apm->echo_cancellation()->GetDelayMetrics(&median, &std,
                                                &fraction_poor_delays);
      printf("\n--Delay metrics--\n");
      printf("Median:             %3d\n", median);
      printf("Standard deviation: %3d\n", std);
      printf("Poor delay values:  %3.1f%%\n", fraction_poor_delays * 100);
    }
  }*/

  return 0;
}

