#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <string.h>
#include <string>
#include <exception>
#include "NumCpp.hpp"
#include "NumCpp/NdArray.hpp"
#include <librealsense2/rsutil.h>
#include <exception>
// #include <Shape.hpp>
using namespace cv;
using namespace rs2;
using namespace std;

static cv::Mat frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

// Converts depth frame to a matrix of doubles with distances in meters
static cv::Mat depth_frame_to_meters( const rs2::depth_frame & f )
{
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo( dm, CV_64F );
    dm = dm * f.get_units();
    return dm;
}


short int res_x = 640;
short int res_y = 360;

Mat frame_hsv = Mat::zeros(Size(res_x, res_y), CV_8UC3);
Mat frame_bgr = Mat::zeros(Size(res_x, res_y), CV_8UC3);

Mat display_in = Mat::zeros(Size(res_x, res_y), CV_8UC3);
Mat display_out = Mat::zeros(Size(res_x, res_y), CV_8UC3);
Mat depth_display_mat = Mat::zeros(Size(res_x, res_y), CV_8UC3);

Point2f cen;
float rad;
float jarak_con;
float point3d[3];
float jarak_bola;
float point2d[2];
float height = 0.165;
float x_rs, y_rs, depth_rs;
static Point2f ball_center;
static float ball_radius;

// PARAMETER
#define LIDAR_DISTANCE 5
#define MERGE_CONSTANTA 10
#define MIN_OBS_AREA 1000

// THRESHOLD
int obs_thress[6] = {0, 0, 0, 180, 255, 50};
int ball_thress[6] = {0, 150, 60, 7, 255, 255};

// VECTOR OBS BUFFER AND BALL BUFFER
std::vector<vector<float>> all_obs;
float ball_xyz[3];

float data_object[200];


int main()
{
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::colorizer color_map;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_ANY, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH);
    auto config = pipe.start(cfg);

    // auto depth_sensor = config.get_device().first<rs2::depth_sensor>();
    // auto depth_scale = depth_sensor.get_depth_scale();

    // const int JARAK_TOLERAN = LIDAR_DISTANCE / depth_scale;
    while (true)
    {
        auto frames = pipe.wait_for_frames();
        rs2::align align_to(RS2_STREAM_COLOR);

        auto color_frame = frames.get_color_frame();
        frames = align_to.process(frames);

        rs2::frame depth_display = frames.get_depth_frame().apply_filter(color_map);
        rs2::depth_frame depth = frames.get_depth_frame();

        auto intrinsic = rs2::video_stream_profile(depth.get_profile()).get_intrinsics();
        auto extrinsic = rs2::video_stream_profile(depth.get_profile()).get_extrinsics_to(rs2::video_stream_profile(color_frame.get_profile()));
        auto depth_intrin = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

        frame_bgr = frame_to_mat(color_frame);
        Mat depth_disp(Size(640, 360), CV_8UC3, (void *)depth_display.get_data(), Mat::AUTO_STEP);
        const int width_c = color_frame.as<rs2::video_frame>().get_width();
        const int height_c = color_frame.as<rs2::video_frame>().get_height();
        Mat color_img(Size(width_c, height_c), CV_16UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);

        depth_display_mat = frame_to_mat(depth_display);
        jarak_con = depth.get_distance(cen.x, cen.y);
        jarak_bola = depth.get_distance(ball_center.x, ball_center.y);
        rs2_deproject_pixel_to_point(point3d, &intrinsic, point2d, jarak_bola);

        // OBSTACLE DETECTION
        Mat color_image_hsv;
        Mat buffer_obs;
        Mat obstacle_final;
        Mat ball_threshold;
        cvtColor(frame_bgr, color_image_hsv, COLOR_BGR2HSV);
        inRange(color_image_hsv, Scalar(obs_thress[0], obs_thress[1], obs_thress[2]), Scalar(obs_thress[3], obs_thress[4], obs_thress[5]), buffer_obs);
        morphologyEx(buffer_obs, buffer_obs, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
        erode(buffer_obs, obstacle_final, (15, 15));
        vector<vector<Point>> obs_contours;
        vector<Vec4i> obs_hierarchy;
        findContours(obstacle_final, obs_contours, obs_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        uint32_t asd=3;
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

                rectangle(frame_bgr, pt1, pt2, Scalar(255, 0, 255), 2);
                float center_point[2] = {float(rect.x + (rect.width / 2)), float(rect.y + (rect.height / 2))};
                circle(frame_bgr, Point(center_point[0], center_point[1]), 2, Scalar(255, 255, 255), 2);

                // FIND DISTANCE
                auto obs_distance = depth.get_distance(center_point[0], center_point[1]);
                float real_obs_coordinates[3]={0,0,0};
                // vector<float>real_obs_coordinates;
                // float real_obs_coordinates;
                float x_ = float(center_point[1]);
                float y_ = float(center_point[0]);
                rs2_deproject_pixel_to_point(real_obs_coordinates, &depth_intrin, center_point, obs_distance);

                for(int i=0;i<3;i++){
                    data_object[asd + i] = real_obs_coordinates[i];
                }

                asd +=3;

            }
        }

        // BALL DETECTION
        inRange(color_image_hsv, Scalar(ball_thress[0], ball_thress[1], ball_thress[2]), Scalar(ball_thress[3], ball_thress[4], ball_thress[5]), ball_threshold);
        morphologyEx(ball_threshold, ball_threshold, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));

        vector<vector<Point>> ball_contours;
        vector<Vec4i> ball_hierarchy;
        findContours(ball_threshold, ball_contours, ball_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int largest_contours = 0;
        int ball_contour;
        int largest_contours_index = 0;
        Point2f ball_center;
        float r_ball;
        
        if (ball_contours.size())
        {
            for (int i = 0; i < ball_contours.size(); i++)
            {
                if (contourArea(ball_contours[i]) > largest_contours)
                {
                    largest_contours = contourArea(ball_contours[i]);
                    largest_contours_index = i;
                    minEnclosingCircle(ball_contours[largest_contours_index], ball_center, r_ball);
                }
            }
            minEnclosingCircle(ball_contours[largest_contours_index], ball_center, r_ball);
            circle(frame_bgr, Point(ball_center.x, ball_center.y), int(r_ball), Scalar(0, 255, 255), 2);
            auto ball_distance = depth.get_distance(ball_center.x, ball_center.y) * 100.0;
            cout << "ball distance: " << ball_distance << endl;
            float ball_center_point[2];
            ball_center_point[0] = ball_center.x;
            ball_center_point[1] = ball_center.y;
            rs2_deproject_pixel_to_point(ball_xyz, &intrinsic, ball_center_point, ball_distance);
            data_object[0] = ball_xyz[0];
            data_object[1] = ball_xyz[1];
            data_object[2] = ball_xyz[2];

        }
        imshow("Color", frame_bgr);
        imshow("depth display", depth_display_mat);

        if (waitKey(1) > 0)
        {
            break;
        }

    }
    

}