#ifndef GRAPH_SLAM__GRAPH_SLAM_HPP_
#define GRAPH_SLAM__GRAPH_SLAM_HPP_

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <rclcpp/rclcpp.hpp>

#include <graphslam_utils/lib/kd_tree.hpp>
#include <graphslam_utils/graphslam_utils.hpp>
#include "graphslam_msgs/msg/lidar_frame.hpp"
#include "graphslam_msgs/msg/lidar_frame_array.hpp"
#include "graphslam_msgs/srv/get_pose.hpp"
#include "graphslam_msgs/srv/save_map.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/gicp_omp.h>
#include <pclomp/ndt_omp.h>
#include <fstream>

using PointType = pcl::PointXYZI;

class GraphSLAM : public rclcpp::Node
{
public:
  GraphSLAM(const rclcpp::NodeOptions & node_options);
  ~GraphSLAM() = default;

  bool detect_loop_with_accum_dist(
    const graphslam_msgs::msg::LidarFrame& latest_lidar_frame,
    const graphslam_msgs::msg::LidarFrameArray& lidar_frame_array,
    std::vector<graphslam_msgs::msg::LidarFrame>& candidate_lidar_frames);

  bool detect_loop_with_kd_tree(
    const graphslam_msgs::msg::LidarFrame& latest_lidar_frame,
    const pcl::PointCloud<PointType>::Ptr& lidar_frame_cloud,
    pcl::PointCloud<PointType>::Ptr& nearest_lidar_frame_cloud, int& closest_lidar_frame_id);

  // Gauss-Newton or Least Squares optimization
  gtsam::Values  optimize_with_gauss_newton();
  // Levenberg Marquardt or Bundle Adjustment optimization
  gtsam::Values  optimize_with_dog_leg();
  // ISAM2 (Incremental Smoothing and Mapping 2) optimization
  gtsam::Values  optimize_with_isam2();
  // SPA (Sparse Pose Adjustment) optimization
  gtsam::Values  optimize_with_levenberg_marquardt();

  void updateFramePose(
    const graphslam_msgs::msg::LidarFrame::SharedPtr& msg, 
    const gtsam::Pose3& pose);
  void lidar_frame_callback(const graphslam_msgs::msg::LidarFrame::SharedPtr msg);
  void  optimize();
  void optimization_callback();

  void adjust_pose();

  void update_estimate_path();
  void publish_map();

  pcl::PointCloud<PointType>::Ptr transform_point_cloud(
    const pcl::PointCloud<PointType>::Ptr input_cloud_ptr, const Eigen::Matrix4f transform_matrix);

  pcl::Registration<PointType, PointType>::Ptr get_registration(
    const std::string registration_method);

  bool get_pose_service(
    const graphslam_msgs::srv::GetPose::Request::SharedPtr req,
    graphslam_msgs::srv::GetPose::Response::SharedPtr res);

  bool save_map_service(
    const graphslam_msgs::srv::SaveMap::Request::SharedPtr req,
    graphslam_msgs::srv::SaveMap::Response::SharedPtr res);

private:
  rclcpp::Subscription<graphslam_msgs::msg::LidarFrame>::SharedPtr lidar_frame_subscriber_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr modified_path_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr modified_map_publisher_;
  rclcpp::Publisher<graphslam_msgs::msg::LidarFrameArray>::SharedPtr
    modified_lidar_frame_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr candidate_lidar_frame_publisher_;

  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Service<graphslam_msgs::srv::GetPose>::SharedPtr get_pose_service_;
  rclcpp::Service<graphslam_msgs::srv::SaveMap>::SharedPtr save_map_service_;

  // registration
  pcl::Registration<PointType, PointType>::Ptr registration_;

  // std::shared_ptr<KDTree> kd_tree_;
  pcl::KdTreeFLANN<PointType>::Ptr kd_tree_;

  // voxel grid filtering
  pcl::VoxelGrid<PointType> voxel_grid_;

  gtsam::NonlinearFactorGraph graph_;
  std::shared_ptr<gtsam::ISAM2> optimizer_;
  gtsam::Values initial_estimate_;
  gtsam::Values current_estimate_;
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;

  pcl::PointCloud<PointType>::Ptr lidar_frame_point_;
  graphslam_msgs::msg::LidarFrameArray lidar_frame_array_;
  graphslam_msgs::msg::LidarFrameArray lidar_frame_raw_array_;

  std::mutex optimize_thread_mutex_;
  std::mutex lidar_frame_update_mutex_;

  bool optimization_performed_{false};
  bool is_loop_closed_{false};
  bool is_initialized_lidar_frame_{false};
  int search_lidar_frame_num_;
  double score_threshold_;
  double search_radius_;
  double search_for_candidate_threshold_;
  double accumulate_distance_threshold_;

  bool use_gauss_newton_optimization_;
  bool use_dog_leg_optimization_;
  bool use_isam2_optimization_;
  bool use_levenberg_marquardt_optimization_;
  bool use_detect_loop_with_accum_dist_;
  bool use_detect_loop_with_kd_tree_;
  bool use_detect_loop_with_min_dist_;

  nav_msgs::msg::Path candidate_line_;
};

#endif