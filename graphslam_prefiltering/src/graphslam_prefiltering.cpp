#include "graphslam_prefiltering/graphslam_prefiltering.hpp"

Prefiltering::Prefiltering(const rclcpp::NodeOptions &node_options)
: Node("graphslam_prefiltering", node_options) {
    // Declare that there are following parameters on this node.
    leaf_size_ = this->declare_parameter<double>("leaf_size");
    random_sample_num_ = this->declare_parameter<double>("random_sample_num");
    mean_k_ = this->declare_parameter<int>("mean_k");
    stddev_ = this->declare_parameter<double>("stddev");
    min_x_ = this->declare_parameter<double>("min_x");
    max_x_ = this->declare_parameter<double>("max_x");
    min_y_ = this->declare_parameter<double>("min_y");
    max_y_ = this->declare_parameter<double>("max_y");
    min_z_ = this->declare_parameter<double>("min_z");
    max_z_ = this->declare_parameter<double>("max_z");
    min_distance_cloud_ = this->declare_parameter<double>("min_distance_cloud");
    max_distance_cloud_ = this->declare_parameter<double>("max_distance_cloud");

    lidar_topic_ = this->declare_parameter<std::string>("lidar_topic");

    // Replace the managed objects of pointers with new filter objects.
    // In this case, pointer initialization.
    random_sample_filter_.reset(new pcl::RandomSample<PointType>);
    voxel_grid_filter_.reset(new pcl::VoxelGrid<PointType>);
    outlier_filter_.reset(new pcl::StatisticalOutlierRemoval<PointType>);

    // Create ROS topic publisher named "filtered_points" and a queue depth of 5.
    filter_points_publisher_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_points", 5);

    // Create ROS topic subscriber to "cloud" topic.
    // The QoS (quality of service) profile to pass on to
    // the rmw (ROS 2 middleware) implementation.
    // std::bind() to register a member function as a callback.
    sensor_points_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS().keep_last(5),
        std::bind(&Prefiltering::callback_sensor_points, this, std::placeholders::_1));
}


// ----- CALLBACKS START ----- // 

void Prefiltering::callback_sensor_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convert ROS message to point cloud.
    /**
     * @TODO: Optimize?
     * @see https://cpp-optimizations.netlify.app/pcl_fromros/
     */
    pcl::PointCloud<PointType>::Ptr input_points(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*msg, *input_points);

    // Apply the distance filter
    pcl::PointCloud<PointType>::Ptr dist_filter_points(new pcl::PointCloud<PointType>);
    distance_filter(input_points, dist_filter_points);

    // Apply cropping
    pcl::PointCloud<PointType>::Ptr crop_points(new pcl::PointCloud<PointType>);
    crop(dist_filter_points, crop_points);

    // Apply downsampling
    pcl::PointCloud<PointType>::Ptr voxel_filter_points(new pcl::PointCloud<PointType>);
    downsample(crop_points, voxel_filter_points);

    // Apply the outlier filter
    pcl::PointCloud<PointType>::Ptr outlier_filter_points(new pcl::PointCloud<PointType>);
    outlier_filter(voxel_filter_points, outlier_filter_points);

    // Convert point cloud to ROS message.
    /** 
     * @TODO: Optimize?
     */
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*outlier_filter_points, output_msg);
    output_msg.header = msg->header;

    filter_points_publisher_->publish(output_msg);
}
// ----- CALLBACKS END ----- // 


// ----- FILTER FUNCTIONS START ----- // 
void Prefiltering::crop(
    const pcl::PointCloud<PointType>::Ptr input_points_ptr,
    const pcl::PointCloud<PointType>::Ptr &output_points_ptr) {
    for (const auto &point : input_points_ptr->points) {
        if (
            (min_x_ < point.x and point.x < max_x_) and
            (min_y_ < point.y and point.y < max_y_) and
            (min_z_ < point.z and point.z < max_z_)) {
            // Emplace a new point in the cloud, at the end of the container.
            output_points_ptr->points.emplace_back(point);
        }
    }
}

void Prefiltering::distance_filter(
    const pcl::PointCloud<PointType>::Ptr input_points_ptr,
    const pcl::PointCloud<PointType>::Ptr &output_points_ptr) {
    for (const auto &point : input_points_ptr->points) {
        // Сonvert to Eigen Vector3f object and calculate l^2 norm.
        const double distance = point.getVector3fMap().norm();
        if (min_distance_cloud_ < distance) {
            // Emplace a new point in the cloud, at the end of the container.
            output_points_ptr->points.emplace_back(point);
        }
    }
}

void Prefiltering::downsample(
    const pcl::PointCloud<PointType>::Ptr input_points_ptr,
    const pcl::PointCloud<PointType>::Ptr &output_points_ptr) {
    // Set the voxel grid leaf size.
    voxel_grid_filter_->setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_grid_filter_->setInputCloud(input_points_ptr);
    voxel_grid_filter_->filter(*output_points_ptr);
}

void Prefiltering::random_sampling(
    const pcl::PointCloud<PointType>::Ptr input_points_ptr,
    const pcl::PointCloud<PointType>::Ptr &output_points_ptr) {
    // Set number of indices to be sampled.
    random_sample_filter_->setSample(random_sample_num_);
    random_sample_filter_->setInputCloud(input_points_ptr);
    random_sample_filter_->filter(*output_points_ptr);
}

void Prefiltering::outlier_filter(
    const pcl::PointCloud<PointType>::Ptr input_points_ptr,
    const pcl::PointCloud<PointType>::Ptr &output_points_ptr) {
    outlier_filter_->setInputCloud(input_points_ptr);
    // Set the number of nearest neighbors to use for mean distance estimation.
    outlier_filter_->setMeanK(mean_k_);
    // Set the standard deviation multiplier for the distance threshold calculation.
    outlier_filter_->setStddevMulThresh(stddev_);
    outlier_filter_->filter(*output_points_ptr);
}

// ----- FILTER FUNCTIONS END ----- // 


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(Prefiltering)
