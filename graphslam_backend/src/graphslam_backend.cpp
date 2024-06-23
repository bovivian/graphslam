#include "graphslam_backend/graphslam_backend.hpp"

using namespace graphslam_utils;

GraphSLAM::GraphSLAM(const rclcpp::NodeOptions & node_options)
: Node("graphslam_backend", node_options)
{
  search_radius_ = this->declare_parameter<double>("search_radius");
  score_threshold_ = this->declare_parameter<double>("score_threshold");
  search_for_candidate_threshold_ =
    this->declare_parameter<double>("search_for_candidate_threshold");
  accumulate_distance_threshold_ = this->declare_parameter<double>("accumulate_distance_threshold");
  search_lidar_frame_num_ = this->declare_parameter<int>("search_lidar_frame_num");
  // Declare optimization flags
  use_gauss_newton_optimization_ = this->declare_parameter<bool>("use_gauss_newton_optimization", false);
  use_dog_leg_optimization_ = this->declare_parameter<bool>("use_dog_leg_optimization", false);
  use_isam2_optimization_ = this->declare_parameter<bool>("use_isam2_optimization", false);
  use_levenberg_marquardt_optimization_ = this->declare_parameter<bool>("use_levenberg_marquardt_optimization", false);
  use_detect_loop_with_accum_dist_ = this->declare_parameter<bool>("use_detect_loop_with_accum_dist", false);
  use_detect_loop_with_kd_tree_ = this->declare_parameter<bool>("use_detect_loop_with_kd_tree", false);
  use_detect_loop_with_min_dist_ = this->declare_parameter<bool>("use_detect_loop_with_min_dist", true);


  lidar_frame_subscriber_ = this->create_subscription<graphslam_msgs::msg::LidarFrame>(
    "lidar_frame", 5, std::bind(&GraphSLAM::lidar_frame_callback, this, std::placeholders::_1));

  modified_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("modified_path", 5);
  candidate_lidar_frame_publisher_ =
    this->create_publisher<nav_msgs::msg::Path>("candidate_lidar_frame", 5);
  modified_lidar_frame_publisher_ =
    this->create_publisher<graphslam_msgs::msg::LidarFrameArray>("modified_lidar_frame", 5);
  modified_map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "modified_map", rclcpp::QoS{1}.transient_local());

  get_pose_service_ = this->create_service<graphslam_msgs::srv::GetPose>(
    "get_pose",
    std::bind(
      &GraphSLAM::get_pose_service, this, std::placeholders::_1, std::placeholders::_2));

  save_map_service_ = this->create_service<graphslam_msgs::srv::SaveMap>(
    "save_map",
    std::bind(
      &GraphSLAM::save_map_service, this, std::placeholders::_1, std::placeholders::_2));

  kd_tree_.reset(new pcl::KdTreeFLANN<PointType>());
  lidar_frame_point_.reset(new pcl::PointCloud<PointType>);

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  optimizer_ = std::make_shared<gtsam::ISAM2>(parameters);

  voxel_grid_.setLeafSize(0.5, 0.5, 0.5);

  const std::string registration_method =
    this->declare_parameter<std::string>("registration_method");
  registration_ = get_registration(registration_method);

  gtsam::Vector Vector6(6);
  Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
  prior_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);

  const double rate = declare_parameter<double>("rate");
  timer_ = create_timer(
    this, get_clock(), rclcpp::Rate(rate).period(),
    std::bind(&GraphSLAM::optimization_callback, this));
}

pcl::Registration<PointType, PointType>::Ptr GraphSLAM::get_registration(
  const std::string registration_method)
{
  pcl::Registration<PointType, PointType>::Ptr registration;

  if (registration_method == "FAST_GICP") {
    RCLCPP_INFO_STREAM(get_logger(), "registration: " << registration_method.c_str());
    fast_gicp::FastGICP<PointType, PointType>::Ptr fast_gicp(
      new fast_gicp::FastGICP<PointType, PointType>);

    const int max_iteration = this->declare_parameter<int>("max_iteration");
    const int omp_num_thread = this->declare_parameter<int>("omp_num_thread");
    const int correspondence_randomness = this->declare_parameter<int>("correspondence_randomness");
    const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
    const double max_correspondence_distance =
      this->declare_parameter<double>("max_correspondence_distance");

    fast_gicp->setCorrespondenceRandomness(correspondence_randomness);
    fast_gicp->setMaximumIterations(max_iteration);
    fast_gicp->setTransformationEpsilon(transformation_epsilon);
    fast_gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
    if (0 < omp_num_thread) fast_gicp->setNumThreads(omp_num_thread);

    registration = fast_gicp;
  } else if (registration_method == "NDT_OMP") {
    RCLCPP_INFO_STREAM(get_logger(), "registration: " << registration_method.c_str());
    pclomp::NormalDistributionsTransform<PointType, PointType>::Ptr ndt_omp(
      new pclomp::NormalDistributionsTransform<PointType, PointType>);

    const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
    const double step_size = this->declare_parameter<double>("step_size");
    const double ndt_resolution = this->declare_parameter<double>("ndt_resolution");
    const int max_iteration = this->declare_parameter<int>("max_iteration");
    const int omp_num_thread = this->declare_parameter<int>("omp_num_thread");

    ndt_omp->setTransformationEpsilon(transformation_epsilon);
    ndt_omp->setStepSize(step_size);
    ndt_omp->setResolution(ndt_resolution);
    ndt_omp->setMaximumIterations(max_iteration);
    ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    if (0 < omp_num_thread) ndt_omp->setNumThreads(omp_num_thread);

    registration = ndt_omp;
  } else if (registration_method == "GICP") {
    RCLCPP_INFO_STREAM(get_logger(), "registration: " << registration_method.c_str());
    pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr gicp(
      new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>());

    const double correspondence_distance =
      this->declare_parameter<double>("correspondence_distance");
    const double max_iteration = this->declare_parameter<int>("max_iteration");
    const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
    const double euclidean_fitness_epsilon =
      this->declare_parameter<double>("euclidean_fitness_epsilon");
    const int ransac_iteration = this->declare_parameter<int>("ransac_iteration");
    const int max_optimizer_iteration = this->declare_parameter<int>("max_optimizer_iteration");

    gicp->setMaxCorrespondenceDistance(correspondence_distance);
    gicp->setMaximumIterations(max_iteration);
    gicp->setMaximumOptimizerIterations(max_optimizer_iteration);
    gicp->setTransformationEpsilon(transformation_epsilon);
    gicp->setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
    gicp->setRANSACIterations(ransac_iteration);

    registration = gicp;
  } else if (registration_method == "ICP") {
    pcl::IterativeClosestPoint<PointType, PointType>::Ptr icp(
      new pcl::IterativeClosestPoint<PointType, PointType>());
    icp->setMaxCorrespondenceDistance(30);
    icp->setMaximumIterations(100);
    icp->setTransformationEpsilon(1e-8);
    icp->setEuclideanFitnessEpsilon(1e-6);
    icp->setRANSACIterations(0);

    registration = icp;
  }

  return registration;
}

// --- OPTIMIZATIONS ---
gtsam::Values GraphSLAM::optimize_with_gauss_newton() {
  gtsam::GaussNewtonParams params;
  params.setVerbosity("ERROR");
  params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  params.relativeErrorTol = 1e-9;
  params.absoluteErrorTol = 1e-9;
  params.errorTol = 1e-9;
  params.maxIterations = 100;

  gtsam::GaussNewtonOptimizer optimizer(graph_, initial_estimate_, params);
  auto result = optimizer.optimize();
  return result;
}

gtsam::Values GraphSLAM::optimize_with_dog_leg() {
  gtsam::DoglegParams params;
  params.setVerbosity("ERROR");
  params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  params.relativeErrorTol = 1e-9;
  params.absoluteErrorTol = 1e-9;
  params.errorTol = 1e-9;
  params.maxIterations = 100;

  gtsam::DoglegOptimizer optimizer(graph_, initial_estimate_, params);
  auto result = optimizer.optimize();
  return result;
}

gtsam::Values GraphSLAM::optimize_with_isam2() {
  optimizer_->update(graph_, initial_estimate_);
  optimizer_->update();

  return optimizer_->calculateEstimate();
}

gtsam::Values GraphSLAM::optimize_with_levenberg_marquardt() {
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosityLM("ERROR");
  params.setVerbosity("ERROR");
  params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  params.lambdaInitial = 1e-3;
  params.lambdaFactor = 1.2;
  params.relativeErrorTol = 1e-9;
  params.absoluteErrorTol = 1e-9;
  params.errorTol = 1e-9;
  params.maxIterations = 100;

  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_, params);
  auto result = optimizer.optimize();
  return result;
}

void GraphSLAM::optimize() {
  gtsam::Values result;
  if (graph_.empty() || initial_estimate_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Graph or initial estimate is empty, skipping optimization.");
    current_estimate_ = initial_estimate_; 
    return;
  }

  if (use_gauss_newton_optimization_) {
    result = optimize_with_gauss_newton();
  } else if (use_dog_leg_optimization_) {
    result = optimize_with_dog_leg();
  } else if (use_isam2_optimization_) {
    result = optimize_with_isam2();
  } else if (use_levenberg_marquardt_optimization_) {
    result = optimize_with_levenberg_marquardt();
  }

  if (result.empty()) {
    current_estimate_ = initial_estimate_; 
    return;
  }

  current_estimate_ = result;
}
// --- ---

void GraphSLAM::lidar_frame_callback(const graphslam_msgs::msg::LidarFrame::SharedPtr msg) 
{
  std::lock_guard<std::mutex> lock(lidar_frame_update_mutex_);
  if (!is_initialized_lidar_frame_) is_initialized_lidar_frame_ = true;

  auto lidar_frame_size = lidar_frame_array_.lidarframes.size();
  auto latest_lidar_frame = geometry_pose_to_gtsam_pose(msg->pose);
  if (lidar_frame_array_.lidarframes.empty()) {
    gtsam::Vector Vector6(6);
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, latest_lidar_frame, prior_noise_));
    initial_estimate_.insert(0, latest_lidar_frame);
  } else {
    auto previous_lidar_frame = geometry_pose_to_gtsam_pose(lidar_frame_array_.lidarframes.back().pose);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      lidar_frame_size - 1, lidar_frame_size, previous_lidar_frame.between(latest_lidar_frame),
      prior_noise_));
    initial_estimate_.insert(lidar_frame_size, latest_lidar_frame);
  }

  optimize();

  if(use_isam2_optimization_){
    graph_.resize(0);
    initial_estimate_.clear();
    current_estimate_ = optimizer_->calculateEstimate();
  }

  auto estimated_pose = current_estimate_.at<gtsam::Pose3>(current_estimate_.size() - 1);

  graphslam_msgs::msg::LidarFrame lidar_frame;
  lidar_frame.header = msg->header;
  lidar_frame.cloud = msg->cloud;
  lidar_frame.pose = gtsam_pose_to_geometry_pose(estimated_pose);
  lidar_frame.accum_distance = msg->accum_distance;
  lidar_frame.id = msg->id;
  lidar_frame_array_.lidarframes.emplace_back(lidar_frame);
  modified_lidar_frame_publisher_->publish(lidar_frame_array_);

  PointType lidar_frame_point;
  lidar_frame_point.x = lidar_frame.pose.position.x;
  lidar_frame_point.y = lidar_frame.pose.position.y;
  lidar_frame_point.z = lidar_frame.pose.position.z;
  lidar_frame_point_->points.emplace_back(lidar_frame_point);

  lidar_frame_raw_array_.lidarframes.emplace_back(*msg);

  if (is_loop_closed_) {
    adjust_pose();
    is_loop_closed_ = false;
  }

  publish_map();
  update_estimate_path();
}

// --- Loop Detect ---
bool GraphSLAM::detect_loop_with_accum_dist(
  const graphslam_msgs::msg::LidarFrame& latest_lidar_frame,
  const graphslam_msgs::msg::LidarFrameArray& lidar_frame_array,
  std::vector<graphslam_msgs::msg::LidarFrame>& candidate_lidar_frames)
{
  const rclcpp::Time latest_stamp = latest_lidar_frame.header.stamp;
  const Eigen::Vector3d latest_pose{
    latest_lidar_frame.pose.position.x, latest_lidar_frame.pose.position.y, latest_lidar_frame.pose.position.z};
  const double latest_accum_dist = latest_lidar_frame.accum_distance;

  for (const auto& lidar_frame : lidar_frame_array.lidarframes) {
    if ((latest_accum_dist - lidar_frame.accum_distance) < accumulate_distance_threshold_) {
      continue;
    }

    const Eigen::Vector3d lidar_frame_pose{
      lidar_frame.pose.position.x, lidar_frame.pose.position.y, lidar_frame.pose.position.z};

    const double lidar_frame_dist = (latest_pose - lidar_frame_pose).norm();
    if (lidar_frame_dist < search_for_candidate_threshold_) {
      candidate_lidar_frames.emplace_back(lidar_frame);
    }
  }

  return !candidate_lidar_frames.empty();
}

bool GraphSLAM::detect_loop_with_kd_tree(
  const graphslam_msgs::msg::LidarFrame& latest_lidar_frame,
  const pcl::PointCloud<PointType>::Ptr& lidar_frame_cloud,
  pcl::PointCloud<PointType>::Ptr& nearest_lidar_frame_cloud, int& closest_lidar_frame_id)
{
  kd_tree_->setInputCloud(lidar_frame_cloud);

  std::vector<int> indices;
  std::vector<float> dists;
  PointType latest_lidar_point;
  latest_lidar_point.x = latest_lidar_frame.pose.position.x;
  latest_lidar_point.y = latest_lidar_frame.pose.position.y;
  latest_lidar_point.z = latest_lidar_frame.pose.position.z;
  kd_tree_->radiusSearch(latest_lidar_point, search_radius_, indices, dists);

  closest_lidar_frame_id = -1;
  for (auto indice : indices) {
    double time_difference = (latest_lidar_frame.header.stamp.sec + latest_lidar_frame.header.stamp.nanosec * 1e-9) -
                             (lidar_frame_array_.lidarframes[indice].header.stamp.sec + lidar_frame_array_.lidarframes[indice].header.stamp.nanosec * 1e-9);
    if (time_difference > 30.0) {
      closest_lidar_frame_id = indice;
      break;
    }
  }

  if (closest_lidar_frame_id == -1) {
    return false;
  }

  const int lidar_frame_size = lidar_frame_array_.lidarframes.size();
  for (int idx = -search_lidar_frame_num_; idx <= search_lidar_frame_num_; ++idx) {
    int lidar_frame_cloud_idx = closest_lidar_frame_id + idx;
    if (lidar_frame_cloud_idx < 0 || lidar_frame_cloud_idx >= lidar_frame_size) continue;

    pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(lidar_frame_array_.lidarframes[lidar_frame_cloud_idx].cloud, *tmp_cloud);

    pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
    const Eigen::Matrix4f matrix = geometry_pose_to_matrix(lidar_frame_array_.lidarframes[lidar_frame_cloud_idx].pose);
    transformed_cloud = transform_point_cloud(tmp_cloud, matrix);
    *nearest_lidar_frame_cloud += *transformed_cloud;
  }

  return true;
}
// --- ---

void GraphSLAM::optimization_callback()
{
  if (lidar_frame_array_.lidarframes.empty()) return;

  std::lock_guard<std::mutex> lock(optimize_thread_mutex_);

  const int lidar_frame_size = lidar_frame_array_.lidarframes.size();
  const auto latest_lidar_frame = lidar_frame_array_.lidarframes.back();
  pcl::PointCloud<PointType>::Ptr nearest_lidar_frame_cloud(new pcl::PointCloud<PointType>);

  pcl::PointCloud<PointType>::Ptr latest_lidar_frame_cloud(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(latest_lidar_frame.cloud, *latest_lidar_frame_cloud);
  pcl::PointCloud<PointType>::Ptr transformed_lidar_frame_cloud(new pcl::PointCloud<PointType>);
  const Eigen::Matrix4f matrix = geometry_pose_to_matrix(latest_lidar_frame.pose);
  transformed_lidar_frame_cloud = transform_point_cloud(latest_lidar_frame_cloud, matrix);

  Eigen::Matrix4f correct_frame;

  double min_dist = std::numeric_limits<double>::max();
  int min_id = -1;

  if (use_detect_loop_with_accum_dist_) {
    std::vector<graphslam_msgs::msg::LidarFrame> candidate_lidar_frames;
    detect_loop_with_accum_dist(latest_lidar_frame, lidar_frame_array_, candidate_lidar_frames);
  } else if (use_detect_loop_with_kd_tree_) {
    pcl::PointCloud<PointType>::Ptr lidar_frame_cloud(new pcl::PointCloud<PointType>);
    for (const auto& lidar_frame : lidar_frame_array_.lidarframes) {
      PointType point;
      point.x = lidar_frame.pose.position.x;
      point.y = lidar_frame.pose.position.y;
      point.z = lidar_frame.pose.position.z;
      lidar_frame_cloud->points.push_back(point);
    }
    int closest_lidar_frame_id;
    detect_loop_with_kd_tree(latest_lidar_frame, lidar_frame_cloud, nearest_lidar_frame_cloud, closest_lidar_frame_id);
  } else if (use_detect_loop_with_min_dist_) {
    const Eigen::Vector3d latest_pose{
      latest_lidar_frame.pose.position.x, latest_lidar_frame.pose.position.y, latest_lidar_frame.pose.position.z};
    const double latest_accum_dist = latest_lidar_frame.accum_distance;

    for (int id = 0; id < lidar_frame_size; id++) {
      auto lidar_frame = lidar_frame_array_.lidarframes[id];
      if ((latest_accum_dist - lidar_frame.accum_distance) < accumulate_distance_threshold_) {
        continue;
      }

      const Eigen::Vector3d lidar_frame_pose{
        lidar_frame.pose.position.x, lidar_frame.pose.position.y, lidar_frame.pose.position.z};

      const double lidar_frame_dist = (latest_pose - lidar_frame_pose).norm();
      if (lidar_frame_dist < search_for_candidate_threshold_) {
        if (lidar_frame_dist < min_dist) {
          min_dist = lidar_frame_dist;
          min_id = id;
        }
      }
    }
  }

  if (min_id == -1) return;

  {
    geometry_msgs::msg::PoseStamped pose_to;
    pose_to.pose = latest_lidar_frame.pose;
    pose_to.header = latest_lidar_frame.header;
    geometry_msgs::msg::PoseStamped pose_from;
    pose_from.pose = lidar_frame_array_.lidarframes[min_id].pose;
    pose_from.header = lidar_frame_array_.lidarframes[min_id].header;
    candidate_line_.poses.emplace_back(pose_from);
    candidate_line_.poses.emplace_back(pose_to);
    candidate_line_.header.frame_id = "map";
    candidate_line_.header.stamp = latest_lidar_frame.header.stamp;
  }

  for (int idx = -search_lidar_frame_num_; idx <= search_lidar_frame_num_; idx++) {
    int lidar_frame_cloud_idx = min_id + idx;
    if (lidar_frame_cloud_idx < 0 or lidar_frame_size <= lidar_frame_cloud_idx) continue;

    pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(lidar_frame_array_.lidarframes[lidar_frame_cloud_idx].cloud, *tmp_cloud);

    pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
    const Eigen::Matrix4f matrix =
      geometry_pose_to_matrix(lidar_frame_array_.lidarframes[lidar_frame_cloud_idx].pose);
    transformed_cloud = transform_point_cloud(tmp_cloud, matrix);
    *nearest_lidar_frame_cloud += *transformed_cloud;
  }

  pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
  voxel_grid_.setInputCloud(nearest_lidar_frame_cloud);
  voxel_grid_.filter(*tmp_cloud);

  registration_->setInputTarget(tmp_cloud);
  registration_->setInputSource(transformed_lidar_frame_cloud);
  pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);
  registration_->align(*output_cloud);

  const Eigen::Matrix4f transform = registration_->getFinalTransformation();
  const double fitness_score = registration_->getFitnessScore();
  const bool has_converged = registration_->hasConverged();

  RCLCPP_INFO_STREAM(get_logger(), "fitness score: " << fitness_score);
  RCLCPP_INFO_STREAM(get_logger(), "min_id: " << min_id);
  candidate_lidar_frame_publisher_->publish(candidate_line_);

  if (!has_converged or score_threshold_ < fitness_score) return;

  // correct position
  auto pose_from = geometry_pose_to_gtsam_pose(
    convert_matrix_to_pose(transform * geometry_pose_to_matrix(latest_lidar_frame.pose)));
  // candidate position
  auto pose_to = geometry_pose_to_gtsam_pose(lidar_frame_array_.lidarframes[min_id].pose);
  gtsam::Vector Vector6(6);
  Vector6 << fitness_score, fitness_score, fitness_score, fitness_score, fitness_score,
    fitness_score;
  gtsam::noiseModel::Diagonal::shared_ptr optimize_noise =
    gtsam::noiseModel::Diagonal::Variances(Vector6);
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
    lidar_frame_size - 1, min_id, pose_from.between(pose_to), optimize_noise));

  RCLCPP_INFO_STREAM(get_logger(), "optimized...");

  is_loop_closed_ = true;
}

pcl::PointCloud<PointType>::Ptr GraphSLAM::transform_point_cloud(
  const pcl::PointCloud<PointType>::Ptr input_cloud_ptr, const Eigen::Matrix4f transform_matrix)
{
  pcl::PointCloud<PointType>::Ptr transform_cloud_ptr(new pcl::PointCloud<PointType>);
  pcl::transformPointCloud(*input_cloud_ptr, *transform_cloud_ptr, transform_matrix);

  return transform_cloud_ptr;
}

void GraphSLAM::adjust_pose()
{
  auto current_estimate_ = optimizer_->calculateEstimate();
  auto estimated_pose = current_estimate_.at<gtsam::Pose3>(current_estimate_.size() - 1);

  for (std::size_t idx = 0; idx < current_estimate_.size(); idx++) {
    lidar_frame_array_.lidarframes[idx].pose =
      gtsam_pose_to_geometry_pose(current_estimate_.at<gtsam::Pose3>(idx));

    PointType lidar_frame_point;
    lidar_frame_point.x = lidar_frame_array_.lidarframes[idx].pose.position.x;
    lidar_frame_point.y = lidar_frame_array_.lidarframes[idx].pose.position.y;
    lidar_frame_point.z = lidar_frame_array_.lidarframes[idx].pose.position.z;
    lidar_frame_point_->points[idx] = lidar_frame_point;
  }
}

void GraphSLAM::update_estimate_path()
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.header.stamp = now();
  for (auto & lidar_frame : lidar_frame_array_.lidarframes) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = lidar_frame.header;
    pose_stamped.pose = lidar_frame.pose;
    path.poses.emplace_back(pose_stamped);
  }
  modified_path_publisher_->publish(path);
}

void GraphSLAM::publish_map()
{
  pcl::PointCloud<PointType>::Ptr map(new pcl::PointCloud<PointType>);
  for (std::size_t idx = 0; idx < lidar_frame_array_.lidarframes.size(); idx++) {
    pcl::PointCloud<PointType>::Ptr lidar_frame_cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(lidar_frame_array_.lidarframes[idx].cloud, *lidar_frame_cloud);

    pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
    const Eigen::Matrix4f matrix = geometry_pose_to_matrix(lidar_frame_array_.lidarframes[idx].pose);
    transformed_cloud = transform_point_cloud(lidar_frame_cloud, matrix);

    *map += *transformed_cloud;
  }

  sensor_msgs::msg::PointCloud2 map_msg;
  pcl::toROSMsg(*map, map_msg);
  map_msg.header.frame_id = "map";
  map_msg.header.stamp = now();
  modified_map_publisher_->publish(map_msg);
}

bool GraphSLAM::get_pose_service(
  const graphslam_msgs::srv::GetPose::Request::SharedPtr req __attribute__((unused)), //req not used, ignore warning
  graphslam_msgs::srv::GetPose::Response::SharedPtr res)
{
  if(lidar_frame_array_.lidarframes.empty())
    return false;

  const auto latest_lidar_frame = lidar_frame_array_.lidarframes.back();
  geometry_msgs::msg::Pose pose = latest_lidar_frame.pose;
  res->pose = pose;

  return true;
}

bool GraphSLAM::save_map_service(
  const graphslam_msgs::srv::SaveMap::Request::SharedPtr req,
  graphslam_msgs::srv::SaveMap::Response::SharedPtr res)
{
  if(lidar_frame_array_.lidarframes.empty()) {
    return false;
  }
  
  pcl::PointCloud<PointType>::Ptr map(new pcl::PointCloud<PointType>);
  for (std::size_t idx = 0; idx < lidar_frame_array_.lidarframes.size(); idx++) {
    pcl::PointCloud<PointType>::Ptr lidar_frame_cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(lidar_frame_array_.lidarframes[idx].cloud, *lidar_frame_cloud);

    pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
    const Eigen::Matrix4f matrix = geometry_pose_to_matrix(lidar_frame_array_.lidarframes[idx].pose);
    transformed_cloud = transform_point_cloud(lidar_frame_cloud, matrix);

    *map += *transformed_cloud;
  }

  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>);

  if (req->resolution <= 0.0) {
    map_cloud = map;
  } else {
    pcl::VoxelGrid<PointType> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(req->resolution, req->resolution, req->resolution);
    voxel_grid_filter.setInputCloud(map);
    voxel_grid_filter.filter(*map_cloud);
  }

  map_cloud->header.frame_id = "map";

  std::ofstream file(req->path);
  if (!file.is_open()) {
      std::cerr << "Failed to create file " << req->path << std::endl;
      return -1;
  }
  file.close();

  int ret = pcl::io::savePCDFile(req->path, *map_cloud);
  res->ret = (ret == 0);

  return true;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(GraphSLAM)
