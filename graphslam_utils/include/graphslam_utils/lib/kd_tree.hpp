#ifndef GRAPH_SLAM__KD_TREE_HPP_
#define GRAPH_SLAM__KD_TREE_HPP_

#include <geometry_msgs/msg/pose.hpp>
#include <graphslam_msgs/msg/lidar_frame.hpp>
#include <graphslam_msgs/msg/lidar_frame_array.hpp>

struct Node
{
  int axis;
  int idx;
  Node * right;
  Node * left;
  std::vector<double> median;
  Node()
  {
    axis = -1;
    idx = -1;
    right = nullptr;
    left = nullptr;
  }
};

typedef std::vector<double> Vector3;

class KDTree
{
public:
  KDTree() {}
  ~KDTree() = default;

  void set_input_cloud(const graphslam_msgs::msg::LidarFrameArray key_frame_array)
  {
    target_.clear();
    for (auto key_frame : key_frame_array.lidarframes) {
      Vector3 point;
      point = {key_frame.pose.position.x, key_frame.pose.position.y, key_frame.pose.position.z};
      target_.emplace_back(point);
    }

    indices_.resize(key_frame_array.lidarframes.size());
    std::iota(indices_.begin(), indices_.end(), 0);

    node_ = build(0, key_frame_array.lidarframes.size() - 1, 0);
  }

  Node * build(int l, int r, int depth)
  {
    if (r <= l) return nullptr;

    int median = (l + r) >> 1;
    int axis = depth % 3;
    // TODO index sort
    std::sort(target_.begin() + l, target_.begin() + r, [&](const auto lhs, const auto rhs) {
      return lhs[axis] < rhs[axis];
    });

    Node * node = new Node();
    node->axis = axis;
    node->idx = median;
    node->median = target_[median];
    node->left = build(l, median, depth + 1);
    node->right = build(median + 1, r, depth + 1);

    return node;
  }

  double calc_euclidean_distance(const Vector3 p1, const Vector3 p2)
  {
    double dist = 0.0;
    for (std::size_t idx = 0; idx < target_.begin()->size(); idx++)
      dist += ((p1[idx] - p2[idx]) * (p1[idx] - p2[idx]));
    return std::sqrt(dist);
  }

  std::vector<Vector3> radius_search(
    const geometry_msgs::msg::Pose pose, const double radius, std::vector<int> & indices)
  {
    Vector3 query{pose.position.x, pose.position.y, pose.position.z};
    std::vector<Vector3> radius_points;
    radius_search_recursive(query, radius, node_, indices, radius_points);
    return radius_points;
  }
  void radius_search_recursive(
    const Vector3 query, const double radius, Node * node, std::vector<int> & indices,
    std::vector<Vector3> & radius_points)
  {
    if (node == nullptr) return;

    const double distance = calc_euclidean_distance(node->median, query);
    if (distance < radius) {
      radius_points.push_back(node->median);
      indices.emplace_back(node->idx);
    }

    Node * next;
    if (query[node->axis] < node->median[node->axis]) {
      radius_search_recursive(query, radius, node->left, indices, radius_points);
      next = node->right;
    } else {
      radius_search_recursive(query, radius, node->right, indices, radius_points);
      next = node->left;
    }

    const double axis_diff = std::fabs(query[node->axis] - node->median[node->axis]);
    if (axis_diff < radius) {
      radius_search_recursive(query, radius, next, indices, radius_points);
    }

    return;
  }

private:
  Node * node_;
  std::vector<Vector3> target_;
  std::vector<int> indices_;
};

#endif
