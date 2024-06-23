#include "graphslam_scan_matcher/graphslam_scan_matcher.hpp"

using namespace graphslam_utils;

ScanMatcher::ScanMatcher(const rclcpp::NodeOptions & node_options)
: Node("graphslam_scan_matcher_node", node_options) {
    // Declare the following parameters on this node.
    base_frame_id_ = this->declare_parameter<std::string>("base_frame_id");
    displacement_ = this->declare_parameter<double>("displacement");
    max_scan_accumulate_num_ = this->declare_parameter<int>("max_scan_accumulate_num");
    use_imu_ = this->declare_parameter<bool>("use_imu");

    // And declare the registration method.
    const std::string registration_method = this->declare_parameter<std::string>("registration_method");

    if (registration_method == "NDT_OMP") {
        // Create the method object.
        pclomp::NormalDistributionsTransform<PointType, PointType>::Ptr ndt_omp(
            new pclomp::NormalDistributionsTransform<PointType, PointType>);

        // Declare the following method parameters on this node.
        const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
        const double step_size = this->declare_parameter<double>("step_size");
        const double ndt_resolution = this->declare_parameter<double>("ndt_resolution");
        const int max_iteration = this->declare_parameter<int>("max_iteration");
        const int omp_num_thread = this->declare_parameter<int>("omp_num_thread");

        // Set parameters for the method.
        ndt_omp->setTransformationEpsilon(transformation_epsilon);
        ndt_omp->setStepSize(step_size);
        ndt_omp->setResolution(ndt_resolution);
        ndt_omp->setMaximumIterations(max_iteration);
        ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        if (0 < omp_num_thread) ndt_omp->setNumThreads(omp_num_thread);

        // Set the method to the class variable.
        registration_ = ndt_omp;

    } else if (registration_method == "FAST_GICP") {
        // Create the method object.
        fast_gicp::FastGICP<PointType, PointType>::Ptr fast_gicp(
            new fast_gicp::FastGICP<PointType, PointType>);

        // Declare the following method parameters on this node.
        const int max_iteration = this->declare_parameter<int>("max_iteration");
        const int omp_num_thread = this->declare_parameter<int>("omp_num_thread");
        const int correspondence_randomness = this->declare_parameter<int>("correspondence_randomness");
        const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
        const double max_correspondence_distance =
            this->declare_parameter<double>("max_correspondence_distance");

        // Set parameters for the method.
        fast_gicp->setCorrespondenceRandomness(correspondence_randomness);
        fast_gicp->setMaximumIterations(max_iteration);
        fast_gicp->setTransformationEpsilon(transformation_epsilon);
        fast_gicp->setMaxCorrespondenceDistance(max_correspondence_distance);
        if (0 < omp_num_thread) fast_gicp->setNumThreads(omp_num_thread);

        // Set the method to the class variable.
        registration_ = fast_gicp;

    } else if (registration_method == "GICP") {
        // Create the method object.
        pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr gicp(
            new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>());

        // Declare the following method parameters on this node.
        const double correspondence_distance =
            this->declare_parameter<double>("correspondence_distance");
        const double max_iteration = this->declare_parameter<int>("max_iteration");
        const double transformation_epsilon = this->declare_parameter<double>("transformation_epsilon");
        const double euclidean_fitness_epsilon =
            this->declare_parameter<double>("euclidean_fitness_epsilon");
        const int max_optimizer_iteration = this->declare_parameter<int>("max_optimizer_iteration");
        const bool use_reciprocal_correspondences =
            this->declare_parameter<bool>("use_reciprocal_correspondences");
        const int correspondence_randomness = this->declare_parameter<int>("correspondence_randomness");

        // Set parameters for the method.
        gicp->setMaxCorrespondenceDistance(correspondence_distance);
        gicp->setMaximumIterations(max_iteration);
        gicp->setUseReciprocalCorrespondences(use_reciprocal_correspondences);
        gicp->setMaximumOptimizerIterations(max_optimizer_iteration);
        gicp->setTransformationEpsilon(transformation_epsilon);
        gicp->setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
        gicp->setCorrespondenceRandomness(correspondence_randomness);

        // Set the method to the class variable.
        registration_ = gicp;
    }

    // Initialize the transform broadcaster.
    broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Create ROS topic publishers.
    //   With durability transient local: the publisher becomes responsible 
    //   for persisting samples for “late-joining” subscriptions.
    front_end_map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "local_map", rclcpp::QoS{1}.transient_local());
    //   With a queue depth of 5.
    scan_match_path_publisher_ =
        this->create_publisher<nav_msgs::msg::Path>("scan_match_path", 5);
    scan_match_pose_publisher_ =
        this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("scan_match_pose", 5);
    scan_match_odom_publisher_ =
        this->create_publisher<nav_msgs::msg::Odometry>("scan_match_odom", 5);
    lidar_frame_publisher_ =
        this->create_publisher<graphslam_msgs::msg::LidarFrame>("lidar_frame", 5);

    // Create ROS topic subscriber to "filtered_points" topic.
    // The QoS (quality of service) profile to pass on to
    // the rmw (ROS 2 middleware) implementation.
    // std::bind() to register a member function as a callback.
    sensor_points_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "filtered_points", rclcpp::SensorDataQoS().keep_last(1),
        std::bind(&ScanMatcher::callback_cloud, this, std::placeholders::_1));
    imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu", 10, std::bind(&ScanMatcher::callback_imu, this, std::placeholders::_1));
}

Eigen::Vector3d ScanMatcher::correct_imu(Eigen::Vector3d  pose_vector, const rclcpp::Time stamp) {
  if (!imu_queue_.empty()) {
    // Get latest imu data
    sensor_msgs::msg::Imu latest_imu_msgs;
    for (auto & imu : imu_queue_) {
      latest_imu_msgs = imu;
      const auto time_stamp = latest_imu_msgs.header.stamp;
      if (stamp < time_stamp) {
        break;
      }
    }
    while (!imu_queue_.empty()) {
      if (rclcpp::Time(imu_queue_.front().header.stamp) >= stamp) {
        break;
      }
      imu_queue_.pop_front();
    }

    static rclcpp::Time previous_stamp = stamp;
    static sensor_msgs::msg::Imu previous_imu = latest_imu_msgs;
    const double dt = (stamp - previous_stamp).seconds();

    imu_rotate_vec_.x() +=
      (latest_imu_msgs.angular_velocity.x - previous_imu.angular_velocity.x) * dt;
    imu_rotate_vec_.y() +=
      (latest_imu_msgs.angular_velocity.y - previous_imu.angular_velocity.y) * dt;
    imu_rotate_vec_.z() +=
      (latest_imu_msgs.angular_velocity.z - previous_imu.angular_velocity.z) * dt;

    pose_vector[3] += imu_rotate_vec_.x();
    pose_vector[4] += imu_rotate_vec_.y();
    pose_vector[5] += imu_rotate_vec_.z();

    previous_imu = latest_imu_msgs;
    previous_stamp = stamp;
  }
  return pose_vector;
}

void ScanMatcher::callback_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<PointType>::Ptr input_cloud_ptr(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr base_to_sensor_cloud(new pcl::PointCloud<PointType>);
    
    // Convert ROS message to point cloud.
    pcl::fromROSMsg(*msg, *input_cloud_ptr);

    // Get the transform between two frames by frame ID.
    const geometry_msgs::msg::TransformStamped base_to_sensor_transform =
        get_transform(msg->header.frame_id, base_frame_id_);
    
    // Apply a rigid transform.
    base_to_sensor_cloud = transform_point_cloud(input_cloud_ptr, base_to_sensor_transform);
    const rclcpp::Time current_scan_time = msg->header.stamp;

    if (!target_cloud_) {
        // Writes the identity expression (matrix) (not necessarily square).
        prev_translation_.setIdentity();
        lidar_frame_.setIdentity();
        // Replace the managed objects of pointers with new filter objects.
        // In this case, pointer initialization.
        target_cloud_.reset(new pcl::PointCloud<PointType>);
        target_cloud_->header.frame_id = "map";

        // Create lidar frame msg.
        graphslam_msgs::msg::LidarFrame lidar_frame;
        lidar_frame.header = msg->header;
        lidar_frame.pose = graphslam_utils::convert_matrix_to_pose(prev_translation_);
        lidar_frame.id = id_;

        id_++;

        // Convert point cloud to ROS message.
        pcl::toROSMsg(*input_cloud_ptr, lidar_frame.cloud);
        // Emplace a new frame msg at the end of the frames msg array.
        lidar_frame_array_.lidarframes.emplace_back(lidar_frame);

        // ...
        *target_cloud_ += *base_to_sensor_cloud;
        // Provide a pointer to the input target
        // (e.g., the point cloud that we want to align the input source to)
        registration_->setInputTarget(target_cloud_);

        // Create local map msg.
        sensor_msgs::msg::PointCloud2 local_map;
        local_map.header.frame_id = "map";
        local_map.header.stamp = current_scan_time;
        // Convert point cloud to ROS message.
        pcl::toROSMsg(*target_cloud_, local_map);
        // Publish message on the topic associated with this publisher.
        front_end_map_publisher_->publish(local_map);
        publish_lidar_frame(lidar_frame);

        return;
    }

    // Provide a pointer to the input source
    // (e.g., the point cloud that we want to align to the target)
    registration_->setInputSource(base_to_sensor_cloud);

    pcl::PointCloud<PointType>::Ptr aligned_cloud_ptr(new pcl::PointCloud<PointType>);
    // Call the registration algorithm which estimates the transformation
    // and returns the transformed source (input) as output.
    registration_->align(*aligned_cloud_ptr, prev_translation_);

    // Return the state of convergence after the last align run.
    if (!registration_->hasConverged()) {
        RCLCPP_ERROR(get_logger(), "LiDAR Scan Matching has not Converged.");
        return;
    }

    // Get the final transformation matrix estimated by the registration method.
    translation_ = registration_->getFinalTransformation();
    pcl::PointCloud<PointType>::Ptr transform_cloud_ptr(new pcl::PointCloud<PointType>);
    // Apply a rigid transform.
    transform_cloud_ptr = transform_point_cloud(
        input_cloud_ptr,
        translation_ * graphslam_utils::convert_transform_to_matrix(base_to_sensor_transform));

    prev_translation_ = translation_;

    const Eigen::Vector3d current_position = translation_.block<3, 1>(0, 3).cast<double>();
    const Eigen::Vector3d previous_position = lidar_frame_.block<3, 1>(0, 3).cast<double>();
    const double delta = (current_position - previous_position).norm();
    
    if (displacement_ <= delta) {
        lidar_frame_ = translation_;
        accum_distance_ += delta;

        target_cloud_->points.clear();

        graphslam_msgs::msg::LidarFrame lidar_frame;
        lidar_frame.header = msg->header;
        lidar_frame.pose = graphslam_utils::convert_matrix_to_pose(lidar_frame_);
        lidar_frame.accum_distance = accum_distance_;
        lidar_frame.id = id_;
        id_++;

        pcl::toROSMsg(*base_to_sensor_cloud, lidar_frame.cloud);
        lidar_frame_array_.lidarframes.emplace_back(lidar_frame);

        const int sub_map_size = lidar_frame_array_.lidarframes.size();
        for (int idx = 0; idx < max_scan_accumulate_num_; idx++) {
            if ((sub_map_size - 1 - idx) < 0) continue;
            pcl::PointCloud<PointType>::Ptr lidar_frame_cloud(new pcl::PointCloud<PointType>);
            pcl::fromROSMsg(lidar_frame_array_.lidarframes[sub_map_size - 1 - idx].cloud, *lidar_frame_cloud);

            pcl::PointCloud<PointType>::Ptr transformed_lidar_cloud(new pcl::PointCloud<PointType>);
            const Eigen::Matrix4f matrix =
            geometry_pose_to_matrix(lidar_frame_array_.lidarframes[sub_map_size - 1 - idx].pose);
            transformed_lidar_cloud = transform_point_cloud(lidar_frame_cloud, matrix);
            *target_cloud_ += *transformed_lidar_cloud;
        }

        registration_->setInputTarget(target_cloud_);

        sensor_msgs::msg::PointCloud2 local_map;
        local_map.header.frame_id = "map";
        local_map.header.stamp = current_scan_time;
        pcl::toROSMsg(*target_cloud_, local_map);
        front_end_map_publisher_->publish(local_map);

        publish_lidar_frame(lidar_frame);
    }

    Eigen::Vector3d translation = translation_.block<3, 1>(0, 3).cast<double>();
    Eigen::Quaterniond quaternion(translation_.block<3, 3>(0, 0).cast<double>());

    if (use_imu_) translation = correct_imu(translation, current_scan_time);

    nav_msgs::msg::Odometry odometry;
    odometry.header.frame_id = "odom";
    odometry.child_frame_id = base_frame_id_;
    odometry.header.stamp = current_scan_time;
    odometry.pose.pose.position = tf2::toMsg(translation);
    odometry.pose.pose.orientation = tf2::toMsg(quaternion);
    scan_match_odom_publisher_->publish(odometry);

    geometry_msgs::msg::PoseWithCovarianceStamped pose_with_covariance;
    pose_with_covariance.header.frame_id = "map";
    pose_with_covariance.header.stamp = current_scan_time;
    pose_with_covariance.pose.pose.position = tf2::toMsg(translation);
    pose_with_covariance.pose.pose.orientation = tf2::toMsg(quaternion);
    scan_match_pose_publisher_->publish(pose_with_covariance);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "map";
    pose_stamped.header.stamp = current_scan_time;
    pose_stamped.pose = pose_with_covariance.pose.pose;
    estimated_path_.poses.emplace_back(pose_stamped);
    estimated_path_.header = pose_stamped.header;
    scan_match_path_publisher_->publish(estimated_path_);

    publish_tf(pose_with_covariance.pose.pose, current_scan_time, "map", base_frame_id_);
}

geometry_msgs::msg::TransformStamped ScanMatcher::get_transform(
    const std::string source_frame, const std::string target_frame)
{
    geometry_msgs::msg::TransformStamped frame_transform;

    try {
        // Get the transform between two frames by frame ID.
        frame_transform = tf_buffer_.lookupTransform(
            target_frame, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.5));
    } catch (tf2::TransformException &ex) {
        RCLCPP_ERROR(get_logger(), "%s", ex.what());
        frame_transform.header.stamp = now();
        frame_transform.header.frame_id = source_frame;
        frame_transform.child_frame_id = target_frame;
        // Zeroing when an error occurs.
        frame_transform.transform.translation.x = 0.0;
        frame_transform.transform.translation.y = 0.0;
        frame_transform.transform.translation.z = 0.0;
        frame_transform.transform.rotation.w = 1.0;
        frame_transform.transform.rotation.x = 0.0;
        frame_transform.transform.rotation.y = 0.0;
        frame_transform.transform.rotation.z = 0.0;
    }
    
    return frame_transform;
}

pcl::PointCloud<PointType>::Ptr ScanMatcher::transform_point_cloud(
    const pcl::PointCloud<PointType>::Ptr input_cloud_ptr,
    const geometry_msgs::msg::TransformStamped transform)
{
    pcl::PointCloud<PointType>::Ptr transform_cloud_ptr(new pcl::PointCloud<PointType>);
    // Convert a timestamped transform to the equivalent Eigen data type.
    const Eigen::Affine3d frame_affine = tf2::transformToEigen(transform);
    // Cast matrix to float.
    const Eigen::Matrix4f frame_matrix = frame_affine.matrix().cast<float>();
    // Apply a rigid transform defined by a 4x4 matrix.
    pcl::transformPointCloud(*input_cloud_ptr, *transform_cloud_ptr, frame_matrix);

    return transform_cloud_ptr;
}

pcl::PointCloud<PointType>::Ptr ScanMatcher::transform_point_cloud(
    const pcl::PointCloud<PointType>::Ptr input_cloud_ptr,
    const Eigen::Matrix4f transform_matrix)
{
    pcl::PointCloud<PointType>::Ptr transform_cloud_ptr(new pcl::PointCloud<PointType>);
    // Apply a rigid transform defined by a 4x4 matrix.
    pcl::transformPointCloud(*input_cloud_ptr, *transform_cloud_ptr, transform_matrix);

    return transform_cloud_ptr;
}

void ScanMatcher::publish_lidar_frame(const graphslam_msgs::msg::LidarFrame lidar_frame) {
    // Publish a message on the topic associated with this publisher.
    lidar_frame_publisher_->publish(lidar_frame);
}

void ScanMatcher::publish_tf(
    const geometry_msgs::msg::Pose pose,
    const rclcpp::Time stamp,
    const std::string frame_id,
    const std::string child_frame_id)
{
    // Create timestamped transform msg.
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.frame_id = frame_id;
    transform_stamped.child_frame_id = child_frame_id;
    transform_stamped.header.stamp = stamp;
    transform_stamped.transform.translation.x = pose.position.x;
    transform_stamped.transform.translation.y = pose.position.y;
    transform_stamped.transform.translation.z = pose.position.z;
    transform_stamped.transform.rotation = pose.orientation;

    // Send a timestamped transform msg.
    broadcaster_->sendTransform(transform_stamped);
}

void ScanMatcher::callback_imu(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  if (!use_imu_) return;

  imu_queue_.emplace_back(*msg);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ScanMatcher)
