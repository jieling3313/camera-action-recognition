#include "ros/ros.h"
#include "yolo_ros/TriggerSelection.h" // Service definition
#include "geometry_msgs/PointStamped.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // std::setprecision

int main(int argc, char **argv)
{
    ros::init(argc, argv, "host_client");
    ros::NodeHandle nh;

    ROS_INFO("Waiting for /trigger_selection service...");
    // 連接伺服器（可接收狀態）
    ros::service::waitForService("/trigger_selection");
    ROS_INFO("Service connected.");

    // 創立Service client
    ros::ServiceClient client = nh.serviceClient<yolo_ros::TriggerSelection>("/trigger_selection");
    
    // 創見Publisher（最終選取物件座標）
    ros::Publisher position_pub = nh.advertise<geometry_msgs::PointStamped>("/selected_object/position", 10);

    // 創 object for sendding
    yolo_ros::TriggerSelection srv;

    ROS_INFO("Sending request to server in container...");
    // 呼叫docker中server
    if (client.call(srv))
    {
        // 是否回傳d435i物件辨識後的物件
        if (srv.response.object_names.empty())
        {
            ROS_INFO("Server reports no objects detected.");
            return 0;
        }

        // 顯示所有被辨識的物件
        std::cout << "\n--- Detected Objects ---" << std::endl;
        for (int i = 0; i < srv.response.object_names.size(); ++i)
        {
            std::cout << "  [" << i + 1 << "] " << srv.response.object_names[i]
          << " (X:" << std::fixed << std::setprecision(2) << srv.response.object_positions[i].x
          << ", Y:" << std::fixed << std::setprecision(2) << srv.response.object_positions[i].y
          << ", Z:" << std::fixed << std::setprecision(2) << srv.response.object_positions[i].z << "mm)" << std::endl;
        }
        std::cout << "--------------------" << std::endl;

        // 挑選夾取物
        std::cout << "Enter the number of the object to publish: ";
        int choice_idx;
        std::cin >> choice_idx;
        choice_idx--; // idx 轉換(array)

        /* 確認挑選物件在物件表中 */
        if (choice_idx >= 0 && choice_idx < srv.response.object_names.size())
        {
            // PointStamped message
            geometry_msgs::PointStamped point_msg;
            point_msg.header.stamp = ros::Time::now();
            point_msg.header.frame_id = "camera_color_optical_frame";
            point_msg.point = srv.response.object_positions[choice_idx];

            // Publish the message
            position_pub.publish(point_msg);
            ROS_INFO_STREAM("Published position of object '" << srv.response.object_names[choice_idx] << "' to /selected_object/position");
            // Wait a moment to ensure the message is sent
            ros::Duration(0.5).sleep();
        }
        else
        {
            ROS_WARN("Invalid selection.");
        }
    }
    else
    {
        ROS_ERROR("Service call failed");
        return 1;
    }

    return 0;
}