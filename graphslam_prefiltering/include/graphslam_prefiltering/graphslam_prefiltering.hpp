#ifndef GRAPHSLAM__GRAPHSLAM_PREFILTERING_NODE_HPP_
#define GRAPHSLAM__GRAPHSLAM_PREFILTERING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

using PointType = pcl::PointXYZI;

class Prefiltering : public rclcpp::Node {
    public:
        Prefiltering(const rclcpp::NodeOptions &node_options);

        /**
         * Callback function for sensor_points_subscriber_.
         * Takes a point cloud, filters it and sends the filtered points.
         */
        void callback_sensor_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

        /**
         * The function filters points by cropping all points that are less
         * than min and greater than max on the x, y and z axes.
         */
        void crop(
            const pcl::PointCloud<PointType>::Ptr input_points_ptr,
            const pcl::PointCloud<PointType>::Ptr &output_points_ptr);

        /**
         * The function filters points based on their distance (points that are less
         * than the min_distance_cloud are not added).
         */
        void distance_filter(
            const pcl::PointCloud<PointType>::Ptr input_points_ptr,
            const pcl::PointCloud<PointType>::Ptr &output_points_ptr);

        /**
         * The function uses the VoxelGrid class object and its methods to filter.
         *
         * VoxelGrid assembles a local 3D grid over a given PointCloud, and
         * downsamples + filters the data.
         * @see https://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html
         */
        void downsample(
            const pcl::PointCloud<PointType>::Ptr input_points_ptr,
            const pcl::PointCloud<PointType>::Ptr &output_points_ptr);

        /**
         * The function uses the RandomSample class object and its methods to filter.
         *
         * RandomSample applies a random sampling with uniform probability.
         * @see https://pointclouds.org/documentation/classpcl_1_1_random_sample.html
         */
        void random_sampling(
            const pcl::PointCloud<PointType>::Ptr input_points_ptr,
            const pcl::PointCloud<PointType>::Ptr &output_points_ptr);

        /**
         * The function uses the StatisticalOutlierRemoval class object and its
         * methods to filter.
         *
         * StatisticalOutlierRemoval uses point neighborhood statistics to filter
         * outlier data.
         * @see https://pointclouds.org/documentation/classpcl_1_1_statistical_outlier_removal.html
         */
        void outlier_filter(
            const pcl::PointCloud<PointType>::Ptr input_points_ptr,
            const pcl::PointCloud<PointType>::Ptr &output_points_ptr);

    private:
        /** ROS topic publishers */
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_points_publisher_;
        
        /** ROS topic subscribers */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor_points_subscriber_;

        /** Filters (filtering objects) */
        pcl::RandomSample<PointType>::Ptr random_sample_filter_;
        pcl::VoxelGrid<PointType>::Ptr voxel_grid_filter_;
        pcl::StatisticalOutlierRemoval<PointType>::Ptr outlier_filter_;

        /** Variables used for downsample() */
        double leaf_size_;
        /** Variables used for random_sampling() */
        double random_sample_num_;
        /** Variables used for distance_filter() */
        double min_distance_cloud_;
        double max_distance_cloud_; // this one isn't actually used
        /** Variables used for outlier_filter() */
        int mean_k_;
        double stddev_;
        /** Variables used for crop() */
        double min_x_;
        double max_x_;
        double min_y_;
        double max_y_;
        double min_z_;
        double max_z_;

        /** Subscribers topics */
        std::string lidar_topic_ = "";

};

#endif // GRAPHSLAM__GRAPHSLAM_PREFILTERING_NODE_HPP_
