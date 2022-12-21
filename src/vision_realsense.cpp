// #include <cv-helpers.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <vector>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <string.h>
#include <string>
#include <exception>
#include "NumCpp.hpp"
#include "NumCpp/NdArray.hpp"
#include <librealsense2/rsutil.h> 
// #include <Shape.hpp>
#define USE_FRAME

// PARAMETER
#define LIDAR_DISTANCE 5
#define MERGE_CONSTANTA 10
#define MIN_OBS_AREA 1000

// THRESHOLD
int obs_thress[6] = {0, 0, 0, 180, 255, 50};
int ball_thress[6] = {0, 150, 60, 7, 255, 255};

// VECTOR OBS BUFFER AND BALL BUFFER
std::vector<int> all_obs;
std::vector<int> ball_xyz;

const size_t windowWidth = 480;
const size_t widnowHeight = 270;

using namespace cv;
using namespace rs2;
using namespace std;

// ============================== cv-helpers.hpp library =====================
static cv::Mat frame_to_mat(const rs2::frame &f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

// Converts depth frame to a matrix of doubles with distances in meters
static cv::Mat depth_frame_to_meters(const rs2::depth_frame &f)
{
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    dm = dm * f.get_units();
    return dm;
}

// ============================== cv-helpers.hpp library =====================

int main()
{
    // get camera parameters
    rs2::pipeline pipe;
    rs2::config cfg;
    
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    rs2::pipeline_profile pipeline_profile = pipe.start(cfg);

    rs2::device rs_dev = pipeline_profile.get_device();
    rs2::align align_to(RS2_STREAM_DEPTH);



    namedWindow("Color ", WINDOW_AUTOSIZE);

    auto depth_sensor = pipeline_profile.get_device().first<rs2::depth_sensor>();
    auto depth_scale = depth_sensor.get_depth_scale();

    const int JARAK_TOLERAN = LIDAR_DISTANCE / depth_scale;


    #ifndef USE_FRAME

    while (true)
    {
        cout << "ERRRRRR" << endl;
        frameset data = pipe.wait_for_frames();
        frameset aligned_set = align_to.process(data);
        frame depth = aligned_set.get_depth_frame();
        Mat color_mat = frame_to_mat(aligned_set.get_color_frame());
        cout << "depth: " << depth << endl;
        cout << "color: " << aligned_set.get_color_frame() << endl;



        imshow("Color", color_mat);

        if (waitKey(1) > 0)
        {
            break;
        }
    }

    #endif

    // STREAMING LOOP

    #ifdef USE_FRAME

    while (true)
    {

        cout << "while loop" << endl;

        // Build a Pipeline
        auto frames = pipe.wait_for_frames();
        auto aligned_frames = align_to.process(frames);
        cout << frames << endl;

        // Get EGB and Streo stream from pipeline
        auto aligned_depth_frame = aligned_frames.get_depth_frame();
        auto color_frame = aligned_frames.get_color_frame();
        cout << "aligned depth frame: " << aligned_depth_frame << endl;
        cout << "color frame: " << color_frame << endl;


        auto depth_intrin = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
        cout << "get depth intrint" << endl;


        // Conversion from camera data to matriks picture (aligned depth frame)
        const int width = color_frame.as<rs2::video_frame>().get_width();
        const int height = color_frame.as<rs2::video_frame>().get_height();

        cout << "width depth frame: " << width << endl;
        cout << "width height frame: " << height << endl;


        // Create opencv matriks of size (width, height) from colored depth data
        Mat depth_img(Size(width, height), CV_16UC1, (void *)color_frame.get_data(), Mat::AUTO_STEP);
        depth_img.convertTo(depth_img, CV_8UC1, 15 / 256.0);
        cout << "convert" << endl;


        auto depth_image = nc::asarray<int>(depth_img);
        // auto depth_image = v_depth_img;

        cout << "to array convert" << endl;

        // Conversion from camera data to matriks picture (color frame)
        const int width_c = color_frame.as<rs2::video_frame>().get_width();
        const int height_c = color_frame.as<rs2::video_frame>().get_height();

        Mat color_img(Size(width_c, height_c), CV_16UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);
        color_img.convertTo(color_img, CV_8UC3, 15 / 256.0);


        auto color_image = nc::asarray<int>(color_img);
        // auto color_image = v_color_img;

        // White Buffer for remove background
        auto depth_image_shape_0 = nc::shape(depth_image.row(0));
        auto depth_image_shape_1 = nc::shape(depth_image.row(1));
        auto white_buffer = nc::ones<int>((depth_image_shape_0, depth_image_shape_1, 3)) * 255;

        auto depth_image_3d = nc::stack({depth_image, depth_image, depth_image});
        auto bg_removed = nc::where((depth_image_3d > JARAK_TOLERAN) | (depth_image_3d <= 0), 0, white_buffer);

        // CONVERT TO MAT
        // color_image, bg_removed,
        auto color_image_rows = color_image.numRows();
        auto color_image_cols = color_image.numCols();
        auto mat_color_image = Mat(color_image_rows, color_image_cols, CV_8UC3, color_image.data());

        auto bg_removed_rows = bg_removed.numRows();
        auto bg_removed_cols = bg_removed.numCols();
        auto mat_bg_removed = Mat(bg_removed_rows, bg_removed_cols, CV_8UC3, bg_removed.data());

        // OBSTACLE DETECTION 3D
        Mat color_image_hsv;
        Mat buffer_obs;
        Mat obstacle_final;
        cvtColor(mat_color_image, color_image_hsv, COLOR_BGR2HSV);
        inRange(color_image_hsv, Scalar(obs_thress[0], obs_thress[1], obs_thress[2]), Scalar(obs_thress[3], obs_thress[4], obs_thress[5]), buffer_obs);
        bitwise_and(mat_bg_removed, mat_bg_removed, obstacle_final, buffer_obs);
        // obstacle_final = ; -> should change type to uint8 but can not
        cvtColor(obstacle_final, obstacle_final, COLOR_BGR2GRAY);
        morphologyEx(obstacle_final, obstacle_final, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(MERGE_CONSTANTA, MERGE_CONSTANTA)));
        erode(obstacle_final, obstacle_final, (15, 15));
        vector<vector<Point>> obs_contours;
        vector<Vec4i> obs_hierarchy;
        findContours(obstacle_final, obs_contours, obs_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < obs_contours.size(); i++)
        {
            if (contourArea(obs_contours[i]) > 1000)
            {
                Rect rect = boundingRect(obs_contours[i]);
                Point pt1, pt2;
                pt1.x = rect.x;
                pt1.y = rect.y;
                pt2.x = rect.x + rect.width;
                pt2.y = rect.y + rect.height;

                rectangle(mat_color_image, pt1, pt2, Scalar(255, 0, 255), 2);
                float center_point[2] = {float(rect.x + (rect.width / 2)), float(rect.y + (rect.height / 2))};
                circle(mat_color_image, Point(center_point[0], center_point[1]), 2, Scalar(255, 255, 255), 2);

                // FIND DISTANCE
                auto obs_distance = aligned_depth_frame.get_distance(center_point[0], center_point[1]);
                float real_obs_coordinates;
                float x_ = float(center_point[1]);
                float y_ = float(center_point[0]);
                rs2_deproject_pixel_to_point(&real_obs_coordinates, &depth_intrin, center_point, obs_distance);

                all_obs.push_back(real_obs_coordinates);
            }
        }

        // BALL DETECTION
        Mat ball_threshold;
        inRange(color_image_hsv, Scalar(ball_thress[0], ball_thress[1], ball_thress[2]), Scalar(ball_thress[3], ball_thress[4], ball_thress[5]), ball_threshold);
        bitwise_and(mat_bg_removed, mat_bg_removed, ball_threshold, ball_threshold);
        cvtColor(ball_threshold, ball_threshold, COLOR_BGR2GRAY);
        morphologyEx(ball_threshold, ball_threshold, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));
        vector<vector<Point>> ball_contours;
        vector<Vec4i> ball_hierarchy;
        findContours(ball_threshold, ball_contours, ball_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int temp = 0;
        int ball_contour;
        int index;
        Point2f ball_center;
        float r_ball;
        if (ball_contours.size())
        {
            for (int i = 0; i < ball_contours.size(); i++)
            {
                if (contourArea(ball_contours[i]) > temp)
                {
                    temp = contourArea(ball_contours[i]);
                    index = i;
                }
            }
            minEnclosingCircle(ball_contours[index], ball_center, r_ball);
            circle(mat_color_image, Point(ball_center.x, ball_center.y), int(r_ball), Scalar(0, 255, 255), 2);
            auto ball_distance = aligned_depth_frame.get_distance(ball_center.x, ball_center.y);
            float ball_xyz;
            float ball_center_point[2];
            ball_center_point[0] = ball_center.x;
            ball_center_point[1] = ball_center.y;
            rs2_deproject_pixel_to_point(&ball_xyz, &depth_intrin, ball_center_point, ball_distance);
        }

        imshow("Depth", obstacle_final);
        imshow("Color", mat_color_image);

        cout << "show vidio" << endl;

        all_obs.clear();
        int key = waitKey(1);

        if (key == 27)
        {
            destroyAllWindows();
            break;
        }
    }

    pipe.stop();

    #endif

}
