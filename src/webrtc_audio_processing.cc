/*	09/2017
	shichaog
	This is the main test File. Include noise suppresion, AEC, VAD.

*/
#include <string>
#include <iostream>

#include "webrtc/modules/audio_processing/include/audio_processing.h"
#include "webrtc/modules/interface/module_common_types.h"

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

int usage() {
    std::cout <<
              "Usage: webrtc-audio-process -anc|-agc|-aec value input.wav output.wav [delay_ms echo_in.wav]"
              << std::endl;
    return 1;
}

bool ReadFrame(FILE* file, webrtc::AudioFrame* frame) {
    // The files always contain stereo audio.
    size_t frame_size = frame->samples_per_channel_;
    size_t read_count = fread(frame->data_,
                              sizeof(int16_t),
                              frame_size,
                              file);
    if (read_count != frame_size) {
        // Check that the file really ended.
        EXPECT_NE(0, feof(file));
        return false;  // This is expected.
    }
    return true;
}

bool WriteFrame(FILE* file, webrtc::AudioFrame* frame) {
    // The files always contain stereo audio.
    size_t frame_size = frame->samples_per_channel_;
    size_t read_count = fwrite (frame->data_,
                                sizeof(int16_t),
                                frame_size,
                                file);
    if (read_count != frame_size) {
        return false;  // This is expected.
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 7) {
        return usage();
    }

    bool is_echo_cancel = false;
    int level = -1;
    level = atoi(argv[2]);
	int delay_ms = 0;

    // Usage example, omitting error checking:
    webrtc::AudioProcessing* apm = webrtc::AudioProcessing::Create();

	webrtc::Config config;
	apm->level_estimator()->Enable(true);

	config.Set<webrtc::ExtendedFilter>(new webrtc::ExtendedFilter(true));
	config.Set<webrtc::DelayAgnostic>(new webrtc::DelayAgnostic(true));

	apm->echo_cancellation()->Enable(true);
    apm->echo_cancellation()->enable_metrics(true);
    apm->echo_cancellation()->enable_delay_logging(true);
	apm->set_stream_delay_ms(delay_ms);

	apm->echo_cancellation()->enable_drift_compensation(true);
    apm->echo_cancellation()->set_suppression_level(webrtc::EchoCancellation::kHighSuppression);

	apm->SetExtraOptions(config);

    apm->high_pass_filter()->Enable(true);
    if (std::string(argv[1]) == "-anc") {
        std::cout << "ANC: level " << level << std::endl;
        apm->noise_suppression()->Enable(true);
        switch (level) {
        case 0:
            apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kLow);
            break;
        case 1:
            apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kModerate);
            break;
        case 2:
            apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kHigh);
            break;
        case 3:
            apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kVeryHigh);
            break;
        default:
            apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kVeryHigh);
        }
        apm->voice_detection()->Enable(true);
    } else if (std::string(argv[1]) == "-agc") {
        std::cout << "AGC: model " << level << std::endl;
        apm->gain_control()->Enable(true);
        apm->gain_control()->set_analog_level_limits(0, 255);
        switch (level) {
        case 0:
            apm->gain_control()->set_mode(webrtc::GainControl::kAdaptiveAnalog);
            break;
        case 1:
            apm->gain_control()->set_mode(webrtc::GainControl::kAdaptiveDigital);
            break;
        case 2:
            apm->gain_control()->set_mode(webrtc::GainControl::kFixedDigital);
            break;
        default:
            apm->gain_control()->set_mode(webrtc::GainControl::kAdaptiveAnalog);
        }
    } else if (std::string(argv[1]) == "-aec") {
        std::cout << "AEC: level " << level << std::endl;
        switch (level) {
            case 0:
                apm->echo_cancellation()->set_suppression_level(webrtc::EchoCancellation::kLowSuppression);
                break;
            case 1:
                apm->echo_cancellation()->set_suppression_level(webrtc::EchoCancellation::kModerateSuppression);
                break;
            case 2:
    			apm->echo_cancellation()->set_suppression_level(webrtc::EchoCancellation::kHighSuppression);
        }
    } else if(std::string(argv[1]) == "-vad"){
        std::cout << "AEC: level " << level << std::endl;
		apm->voice_detection()->Enable(true);
		switch(level){
			case 0:
				apm->voice_detection()->set_likelihood(webrtc::VoiceDetection::kVeryLowLikelihood);
      			apm->voice_detection()->set_frame_size_ms(10);
		}
	
    } else {

        delete apm;
        return usage();
    }

  	webrtc::AudioFrame far_frame;
    webrtc::AudioFrame near_frame;

    float frame_step = 10;  // ms
	int near_read_bytes;

	far_frame.sample_rate_hz_ = 16000;
	far_frame.samples_per_channel_ = far_frame.sample_rate_hz_ * frame_step / 1000.0;
	far_frame.num_channels_ = 1;
	near_frame.sample_rate_hz_ = 16000;
	near_frame.samples_per_channel_ = near_frame.sample_rate_hz_ * frame_step / 1000.0;
	near_frame.num_channels_ = 1;

	size_t size = near_frame.samples_per_channel_;

    FILE *far_file = fopen(argv[3], "rb");
    FILE *near_file = fopen(argv[4], "rb");
    FILE *aec_out_file = fopen(argv[5], "wb");
	int drift_samples = 0;

	size_t read_count = 0;
	while(true){
		read_count = fread(far_frame.data_, sizeof(int16_t), size, far_file);
		if(read_count != size){
			fseek(near_file, read_count * sizeof(int16_t), SEEK_CUR);
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
		int vad_flag = (int)apm->voice_detection()->stream_has_voice();


        WriteFrame(aec_out_file, &near_frame);
		
	}

    fclose(far_file);
    fclose(near_file);
    fclose(aec_out_file);

    delete apm;

    return 0;
}
