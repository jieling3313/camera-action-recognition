#include "ros/ros.h"
#include "yolo_ros/TriggerSelection.h" // Service definition
#include "geometry_msgs/PointStamped.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // For std::setprecision

int main(int argc, char **argv)
{
    // Initialize ROS node
    ros::init(argc, argv, "host_client");
    ros::NodeHandle nh;

    ROS_INFO("Waiting for /trigger_selection service...");
    // Wait for the Service to be available
    ros::service::waitForService("/trigger_selection");
    ROS_INFO("Service connected.");

    // Create a Service client
    ros::ServiceClient client = nh.serviceClient<yolo_ros::TriggerSelection>("/trigger_selection");
    
    // Create a Publisher for the final selected object position
    ros::Publisher position_pub = nh.advertise<geometry_msgs::PointStamped>("/selected_object/position", 10);

    // Create a Service object
    yolo_ros::TriggerSelection srv;

    ROS_INFO("Sending request to server in container...");
    // Call the Service
    if (client.call(srv))
    {
        // Check if the server returned any objects
        if (srv.response.object_names.empty())
        {
            ROS_INFO("Server reports no objects detected.");
            return 0;
        }

        // Display the list of selectable objects in the local terminal
        std::cout << "\n--- Detected Objects ---" << std::endl;
        for (int i = 0; i < srv.response.object_names.size(); ++i)
        {
            std::cout << "  [" << i + 1 << "] " << srv.response.object_names[i]
                      << " (Position Z: " << std::fixed << std::setprecision(2) << srv.response.object_positions[i].z << "m)" << std::endl;
        }
        std::cout << "--------------------" << std::endl;

        // Let the user choose
        std::cout << "Enter the number of the object to publish: ";
        int choice_idx;
        std::cin >> choice_idx;
        choice_idx--; // Convert to 0-based index

        if (choice_idx >= 0 && choice_idx < srv.response.object_names.size())
        {
            // Create a PointStamped message
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