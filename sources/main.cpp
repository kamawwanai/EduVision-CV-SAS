#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/string.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <sqlite3.h>
#include <sqlite_orm/sqlite_orm.h>

namespace fs = std::filesystem;

using namespace dlib; //NOLINT
using namespace std; //NOLINT

// Определение модели нейронной сети для извлечения лицевых признаков
template <template <int,template<typename>class,int,typename> class block, int
N, template<typename>class BN, typename SUBNET> using residual =
add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int
N, template<typename>class BN, typename SUBNET> using residual_down =
add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      =
relu<residual<block,N,affine,SUBNET>>; template <int N, typename SUBNET> using
ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 =
ares<256,ares<256,ares_down<256,SUBNET>>>; template <typename SUBNET> using
alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>; template <typename SUBNET>
using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>; template
<typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();


void train_model() {
    anet_type net;
    deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    shape_predictor sp;
    deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;

    std::vector<matrix<rgb_pixel>> faces;
    std::vector<int> labels;

    std::string data_path = "../person_data/";

    for (const auto& entry : fs::directory_iterator(data_path)) {
        if (entry.is_directory()) {
            int label = stoi(entry.path().filename().string());
            std::cout << "Label: " << label << " for directory: " << entry.path().string() << std::endl;
            for (const auto& file : fs::directory_iterator(entry.path())) {
                if (file.path().extension() == ".jpg" || file.path().extension() == ".png") {
                    matrix<rgb_pixel> img;
                    load_image(img, file.path().string());

                    std::vector<rectangle> dets = detector(img);
                    if (dets.size() == 1) {
                        auto shape = sp(img, dets[0]);
                        matrix<rgb_pixel> face_chip;
                        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                        faces.push_back(std::move(face_chip));
                        labels.push_back(label);
                    }
                }
            }
            label++;
        }
    }

    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
    serialize("../models/face_descriptors.dat") << face_descriptors << labels;
}

class Models {
public:
    Models() {
        try { dlib::deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp; }
                catch (const std::exception& e) {
                    std::cout << "Error loading shape_predictor_68_face_landmarks.dat" << std::endl;
                    std::cerr << "Error loading shape_predictor_68_face_landmarks.dat: " << e.what() << std::endl; throw;
        }

        try { dlib::deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net; }
                catch (const std::exception& e) { std::cout << "Error loading dlib_face_recognition_resnet_model_v1.dat" << std::endl;
                std::cerr << "Error loading dlib_face_recognition_resnet_model_v1.dat: " << e.what() << std::endl;
            throw;
        }
    }

    dlib::shape_predictor sp;
    anet_type net;
};

std::mutex frame_mutex;
cv::Mat current_frame;
std::atomic<bool> new_frame_ready(false);
std::atomic<bool> stop(false);
std::condition_variable frame_cond;


void recognize_faces(cv::CascadeClassifier& face_cascade, anet_type& net, dlib::shape_predictor& sp, 
                     std::vector<dlib::matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, 
                     std::vector<std::string>& names) {
    while (!stop) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cond.wait(lock, [] { return new_frame_ready || stop; });

        if (stop) { break; }

        cv::Mat frame = current_frame.clone();
        new_frame_ready = false;
        lock.unlock();

        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(frame.cols / 2, frame.rows / 2)); // Reduce image size for faster processing

        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        for (auto& face : faces) {
            cv::Rect scaled_face(face.x * 2, face.y * 2, face.width * 2, face.height * 2); // Scale back to original size
            cv::Mat face_roi = frame(scaled_face);
            dlib::cv_image<dlib::bgr_pixel> cimg(face_roi);
            auto shape = sp(cimg, dlib::rectangle(0, 0, face_roi.cols, face_roi.rows));
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            dlib::matrix<float, 0, 1> face_descriptor = net(face_chip);

            float min_distance = 0.6;
            int label = -1;
            for (size_t j = 0; j < face_descriptors.size(); ++j) {
                float distance = dlib::length(face_descriptor - face_descriptors[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    label = labels[j];
                }
            }

            std::string name = (label == -1) ? "Unknown" : names[label];
            int x1 = scaled_face.x;
            int y1 = scaled_face.y;

            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                cv::rectangle(current_frame, scaled_face, cv::Scalar(0, 255, 0), 2);
                cv::putText(current_frame, name, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                std::cout << name << std::endl;
            }
        }
    }
}

auto main() -> int {
    // train_model();
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        return -1;
    }

    anet_type net;
    dlib::deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    dlib::shape_predictor sp;
    dlib::deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;

    std::vector<matrix<float, 0, 1>> face_descriptors;
    std::vector<int> labels;
    deserialize("../models/face_descriptors.dat") >> face_descriptors >> labels;

    std::vector<std::string> names{"angeline_jolie", "brad_pitt", "denzel_washington", "hugh_jackman",
                                   "jennifer_lawrence", "johnny_depp", "kate_winslet", "leonardo_dicaprio",
                                   "magan_fox", "natalie_portman", "nicole_kidman", "robert_downey_jr",
                                   "sandra_bullock", "scarlett_johansson", "tom_cruise", "tom_hanks",
                                   "will_smith", "ksenia_karimova", "roma_kislitsyn", "lera_krasovskaya",
                                   "dima_kutuzov"};

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("../models/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading Haar cascade" << std::endl;
        return -1;
    }

    std::thread recognition_thread(recognize_faces, std::ref(face_cascade), std::ref(net), std::ref(sp),
                                   std::ref(face_descriptors), std::ref(labels), std::ref(names));

    while (true) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            cap >> current_frame;
            if (current_frame.empty()) {
                break;
            }
        }

        new_frame_ready = true;
        frame_cond.notify_one();

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            cv::imshow("Face Recognition", current_frame);
        }

        if (cv::waitKey(30) == 27) {
            stop = true;
            frame_cond.notify_one();
            break;
        }
    }

    recognition_thread.join();

    cap.release();
    cv::destroyAllWindows();
    return 0;
}