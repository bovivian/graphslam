#ifndef GRAPHSLAM__GRAPHSLAM_SCAN_MATCHER_NODE_HPP_
#define GRAPHSLAM__GRAPHSLAM_SCAN_MATCHER_NODE_HPP_

#include "graphslam_utils/graphslam_utils.hpp"

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <graphslam_msgs/msg/lidar_frame.hpp>
#include <graphslam_msgs/msg/lidar_frame_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

using PointType = pcl::PointXYZI;

class ScanMatcher : public rclcpp::Node {
    public:
        /**
         * Constructor: declaring node parameters, setting the registration method,
         * initializing the broadcaster, creating publishers and subscribers.
         */
        ScanMatcher(const rclcpp::NodeOptions & node_options);
        ~ScanMatcher() = default;
        
        /**
         * Callback function for sensor_points_subscriber_.
         *
         * TODO: descibe func pls
         *
         * @exception RCLCPP_ERROR if matrix is not convergent after alignment.
         * @see https://en.wikipedia.org/wiki/Convergent_matrix
         */
        void callback_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
        
        /** TODO: Implement.  */
        void callback_imu(const sensor_msgs::msg::Imu::SharedPtr msg);
        void callback_odometry(const nav_msgs::msg::Odometry::SharedPtr msg);

        Eigen::Vector3d correct_imu(Eigen::Vector3d pose_vector, const rclcpp::Time stamp);
        
        pcl::Registration<PointType, PointType>::Ptr get_registration();
        nav_msgs::msg::Odometry convert_to_odometry(const geometry_msgs::msg::Pose pose);

        /**
         * Get the transform between two frames by frame ID.
         *
         * @exception TransformException resets the returned transform to zero.
         * @return the transform between the frames.
         */
        geometry_msgs::msg::TransformStamped get_transform(
            const std::string source_frame, const std::string target_frame);
        
        /**
         * Apply a rigid transform defined by a 4x4 matrix.
         *
         * @param input_cloud_ptr is a pointer to the source point cloud.
         * @param transform is a timestamped transform converted to a 4x4 matrix.
         * @param transform_matrix is a 4x4 matrix.
         * @return Pointer to the transformed cloud.
         */
        pcl::PointCloud<PointType>::Ptr transform_point_cloud(
            const pcl::PointCloud<PointType>::Ptr input_cloud_ptr,
            const geometry_msgs::msg::TransformStamped transform);
        pcl::PointCloud<PointType>::Ptr transform_point_cloud(
            const pcl::PointCloud<PointType>::Ptr input_cloud_ptr,
            const Eigen::Matrix4f transform_matrix);
        
        /**
         * Publish a lidar frame message on the topic associated
         * with lidar_frame_publisher_.
         */
        void publish_lidar_frame(const graphslam_msgs::msg::LidarFrame lidar_frame);

        /**
         * Creates TransformStamped message and broadcasts
         * the transformation from tf frame child to parent
         * on ROS topic /tf.
         */
        void publish_tf(
            const geometry_msgs::msg::Pose pose,
            const rclcpp::Time stamp,
            const std::string frame_id,
            const std::string child_frame_id);

    private:
        /** ROS topic publishers */
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr front_end_map_publisher_;
        rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr scan_match_pose_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr scan_match_odom_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr scan_match_path_publisher_;
        rclcpp::Publisher<graphslam_msgs::msg::LidarFrame>::SharedPtr lidar_frame_publisher_;
        
        /** ROS topic subscribers */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_points_subscriber_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;

        pcl::PointCloud<PointType>::Ptr target_cloud_;

        // registration
        pcl::Registration<PointType, PointType>::Ptr registration_;

        std::string registration_type_;

        graphslam_msgs::msg::LidarFrameArray lidar_frame_array_;

        Eigen::Matrix4f lidar_frame_;
        Eigen::Matrix4f translation_;
        Eigen::Matrix4f prev_translation_;

        geometry_msgs::msg::Pose current_pose_;

        nav_msgs::msg::Path estimated_path_;

        Eigen::Vector3d imu_rotate_vec_;
        std::deque<sensor_msgs::msg::Imu> imu_queue_;

        tf2_ros::Buffer tf_buffer_{get_clock()};
        tf2_ros::TransformListener tf_listener_{tf_buffer_};
        /**
         * TransformBroadcaster is a convenient way to send
         * transformation updates on the /tf message topic.
         */
        std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;

        std::string base_frame_id_ = "";
        std::string sensor_frame_id_ = "";
        // std::string base_frame_id_;
        // std::string sensor_frame_id_;

        int max_scan_accumulate_num_;
        double displacement_;
        double accum_distance_{0.0};
        bool use_imu_{false};

        /** Lidar frame id. */
        int id_{0};
};

#endif // GRAPHSLAM__GRAPHSLAM_SCAN_MATCHER_NODE_HPP_
