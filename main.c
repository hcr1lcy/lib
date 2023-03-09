
#include <stdio.h>
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/port/opencv_core_inc.h>
#include <mediapipe/framework/port/opencv_imgproc_inc.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/solutions/drawing_utils.h>
#include <mediapipe/solutions/face_mesh.h>
#include <mediapipe/util/tracking/region_flow.h>

using mediapipe::CalculatorGraphConfig;
using mediapipe::ParseTextProtoOrDie;
using mediapipe::face_mesh::FaceMesh;
using mediapipe::solutions::drawing_utils::DrawingSpec;
using mediapipe::solutions::drawing_utils::DrawingStyle;

int main() {
  // Initialize CalculatorGraph config
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "FaceMeshCalculator"
          input_stream: "VIDEO:video_frames"
          output_stream: "FACE_MESH:face_mesh"
          options {
            [mediapipe.FaceMeshCalculatorOptions.ext] {
              min_detection_confidence: 0.5
              min_tracking_confidence: 0.5
            }
          }
        }
        node {
          calculator: "VideoToImageCalculator"
          input_stream: "VIDEO:video_frames"
          output_stream: "IMAGE:image_frame"
        }
        node {
          calculator: "DrawFaceMeshCalculator"
          input_stream: "IMAGE:image_frame"
          input_stream: "FACE_MESH:face_mesh"
          output_stream: "IMAGE:image_frame_with_mesh"
          options {
            [mediapipe.DrawFaceMeshCalculatorOptions.ext] {
              landmark_drawing_spec {
                thickness: 1
                circle_radius: 1
              }
              tesselation_style: SOLID
            }
          }
        }
      )");

  // Open video capture
  cv::VideoCapture cap(0);
  cv::Mat image;

  // Initialize FaceMesh
  FaceMesh face_mesh;
  face_mesh.Initialize();

  // Run the Calculator graph
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  MP_RETURN_IF_ERROR(graph.StartRun({{"VIDEO:video_frames", cap}}));

  // Main loop
  while (cap.read(image)) {
    // Flip the image horizontally and convert the color space from BGR to RGB
    image = cv::cvtColor(cv::flip(image, 1), cv::COLOR_BGR2RGB);

    // Detect the face landmarks
    auto results = face_mesh.Process(image);

    // Convert back to the BGR color space
    image = cv::cvtColor(image, cv::COLOR_RGB2BGR);

    // Draw the face mesh annotations on the image
    if (results.multi_face_landmarks.size() > 0) {
      DrawingSpec spec;
      spec.thickness = 1;
      spec.circle_radius = 1;
      mp::solutions::drawing_utils::DrawLandmarks(
          image, results.multi_face_landmarks,
          mp::face_mesh::FACEMESH_TESSELATION, spec,
          mp::solutions::drawing_styles::GetDefaultFaceMeshTesselationStyle());
    }

    // Display the image
    cv::imshow("MediaPipe FaceMesh", image);

    // Terminate the process
    if (cv::waitKey(5) & 0xFF == 27) {
      break;
    }
  }

  // Close the Calculator graph
  MP_RETURN_IF_ERROR(graph.CloseInputStream("VIDEO:video_frames"));
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
}