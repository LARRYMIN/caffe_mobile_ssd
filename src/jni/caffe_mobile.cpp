#include "caffe_mobile.hpp"

namespace caffe {

CaffeMobile *CaffeMobile::caffe_mobile_ = NULL;

CaffeMobile *CaffeMobile::get() {
  return caffe_mobile_;
}

CaffeMobile *CaffeMobile::get(const string &param_file,
                              const string &trained_file) {
  if (!caffe_mobile_) {
    try {
      caffe_mobile_ = new CaffeMobile(param_file, trained_file);
    } catch (std::invalid_argument &e) {
      // TODO
    }
  }
  return caffe_mobile_;
}

CaffeMobile::CaffeMobile(const string &param_file, const string &trained_file) {
  // Load Caffe model
  Caffe::set_mode(Caffe::CPU);

  printf("\n mw caffemobile  \n");
  printf("\n mw param_file = %s \n", &param_file);
  printf("\n mw trained_file = %s \n", &trained_file);

  CPUTimer timer;
  timer.Start();

  printf("\n mw caffemobile ----------0001  \n");

  net_.reset(new Net<float>(param_file, caffe::TEST));
  
  printf("\n mw caffemobile ----------001  \n");
  if (net_.get() == NULL) {
    printf("\n mw caffemobile ----------0 \n");
    throw std::invalid_argument("Invalid arg: param_file=" + param_file);
  }
  printf("\n mw caffemobile ----------01  \n");
  net_->CopyTrainedLayersFrom(trained_file);
  printf("\n mw caffemobile ----------02  \n");
  timer.Stop();
  LOG(INFO) << "Load (" << param_file << "," << trained_file << "), time:"
            << timer.MilliSeconds() << " ms.";

  printf("\n mw caffemobile ----------1  \n");

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  // Get input_layer info
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_channels_ = input_layer->channels();
  CHECK(input_channels_ == 3 || input_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_width_  = input_layer->width();
  input_height_ = input_layer->height();
  printf("\n mw caffemobile ----------2  \n");


}

CaffeMobile::~CaffeMobile() {
  net_.reset();
}

bool CaffeMobile::predictImage(const uint8_t* rgba,
                               int channels,
                               const std::vector<float> &mean,
                               std::vector<float> &result) {
  if ((rgba == NULL) || net_.get() == NULL) {
    LOG(ERROR) << "Invalid arguments: rgba=" << rgba
        << ",net_=" << net_.get();
    return false;
  }
  CPUTimer timer;
  timer.Start();
  // Write input
  Blob<float> *input_layer = net_->input_blobs()[0];
  float *input_data = input_layer->mutable_cpu_data();
  size_t plane_size = input_height() * input_width();
  if (input_channels() == 1 && channels == 1) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = static_cast<float>(rgba[i]);  // Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels() == 1 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]; // RGB2Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels() == 3 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = static_cast<float>(rgba[i * 4 + 2]);                   // B
      input_data[plane_size + i] = static_cast<float>(rgba[i * 4 + 1]);      // G
      input_data[2 * plane_size + i] = static_cast<float>(rgba[i * 4]);      // R
      // Alpha is discarded
      if (mean.size() == 3) {
        input_data[i] -= mean[0];
        input_data[plane_size + i] -= mean[1];
        input_data[2 * plane_size + i] -= mean[2];
      }
    }
  } else {
    LOG(ERROR) << "image_channels input_channels not match.";
    return false;
  }
  // Do Inference
  net_->Forward();
  timer.Stop();
  LOG(INFO) << "Inference use " << timer.MilliSeconds() << " ms.";
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->shape(1);
  result.assign(begin, end);
  return true;
}

} // namespace caffe
