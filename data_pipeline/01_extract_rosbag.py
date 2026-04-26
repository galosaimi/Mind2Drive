#!/usr/bin/env python3
"""
01_extract_rosbag.py
--------------------
Extract and synchronise multi-sensor data from ROS bag files.

Traverses all subject/session directories, reads EEG, GPS, odometry,
camera, and IMU messages, and saves each modality as a separate CSV file.
Images are saved as individual PNG files named by timestamp.

Inputs  : ROS bag files 
Outputs : Per-session CSVs and images 

Author : Ghadah Alosaimi, Durham University | Imam Mohammad Ibn Saud Islamic University 
"""

import rosbag
import numpy as np
import csv
import os
import cv2
from cv_bridge import CvBridge
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import time


# ---------------------------------------------------------------------------
# Configuration: input/output paths and ROS topic names
# ---------------------------------------------------------------------------
base_input_dir = "data/raw_rosbags"
base_output_dir = "data/extracted"
topics = {
"eeg": "/eeg_data",
"gps": "/gps/vel",
"odom": "/gps/odom",
"image": "/multisense/left/image_rect_color",
"imu": "/os_cloud_node/imu"
}

sampling_rate = 125                 # OpenBCI sample rate
default_dt = 1.0 / sampling_rate    # 0.008s

# 16-channel EEG electrode layout (international 10-20 system)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2',
    'F7', 'F8', 'F3', 'F4', 'T5', 'T6', 'P3', 'P4'
]

# ROS image bridge and session log initialisation
bridge = CvBridge()
log_lines = []

# Traverse all subjects and sessions in the input directory
for subject in sorted(os.listdir(base_input_dir)):
    subject_path = os.path.join(base_input_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for filename in sorted(os.listdir(subject_path)):
        if not filename.endswith(".bag"):
            continue

        session_name = os.path.splitext(filename)[0]    # e.g., ST01S01
        print(f"Processing {subject} - {session_name}...")
        bag_path = os.path.join(subject_path, filename)
        output_dir = os.path.join(base_output_dir, subject, session_name)
        os.makedirs(output_dir, exist_ok=True)

        eeg_rows = []
        gps_rows = []
        odom_rows = []
        image_rows = []
        imu_rows = []
        prev_last_timestamp = None
        skipped_messages = 0
        skipped_partial_samples = 0
        total_samples = 0

        image_save_dir = os.path.join(output_dir, "images")
        os.makedirs(image_save_dir, exist_ok=True)

        log_lines.append(f"\nProcessing {session_name}...")
        start_time = time.time()


        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=list(topics.values())):
                timestamp = msg.header.stamp.to_sec()

                if topic == topics["eeg"]:
                    num_channels = len(msg.channels)
                    total_values = len(msg.data)
                    num_samples = total_values // num_channels

                    if num_samples == 0:
                        skipped_messages += 1
                        continue

                    # Handle leftover values
                    expected_values = num_samples * num_channels
                    skipped_values = total_values - expected_values
                    if skipped_values > 0:
                        skipped_partial_samples += skipped_values // num_channels
                        log_lines.append(
                        f"{session_name}: WARNING - dropped {skipped_values} leftover EEG values in one message"
                    )
                    
                    # Reshape data
                    data = np.array(msg.data[:expected_values])
                    eeg_matrix = data.reshape(num_channels, -1).T
                    num_samples = eeg_matrix.shape[0] 
                    
                    # Timestamping
                    msg_time = timestamp
                    if prev_last_timestamp is None:
                        # First message: fixed stepping backward
                        timestamps = [msg_time - (num_samples - 1 - i) * default_dt for i in range(num_samples)]
                    else:
                        estimated_first = msg_time - (num_samples - 1) * default_dt

                        if estimated_first <= prev_last_timestamp:
                            # Overlap: spread real duration between prev_last_timestamp and current msg_time
                            duration = msg_time - prev_last_timestamp
                            dt = duration / num_samples
                            t0 = prev_last_timestamp + dt
                            timestamps = [t0 + i * dt for i in range(num_samples)]
                        else:
                            # Normal case: 125 Hz stepping
                            timestamps = [msg_time - (num_samples - 1 - i) * default_dt for i in range(num_samples)]

                    prev_last_timestamp = timestamps[-1]

                    for i in range(num_samples):
                        row = [timestamps[i]] + eeg_matrix[i].tolist()
                        eeg_rows.append(row)

                    total_samples += num_samples
                    # msg_count += 1
                
                elif topic == topics["gps"]:
                    vel_x = msg.twist.twist.linear.x
                    vel_y = msg.twist.twist.linear.y
                    gps_rows.append([timestamp, vel_x, vel_y])
                    
                elif topic == topics["odom"]:
                    # Yaw from quaternion
                    orientation_q = msg.pose.pose.orientation
                    quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
                    euler = tf.transformations.euler_from_quaternion(quaternion)
                    yaw = euler[2]  # in radians
                    yaw_rate = msg.twist.twist.angular.z
                    odom_rows.append([timestamp, yaw, yaw_rate])

                elif topic == topics["image"]:
                    # Convert ROS Image message to OpenCV format
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    image_filename = f"{timestamp:.9f}.png"
                    image_path = os.path.join(image_save_dir, image_filename)
                    
                    # Save image with timestamp as filename
                    cv2.imwrite(image_path, cv_image)
                    image_rows.append([f"{timestamp:.9f}", image_path])

                elif topic == topics["imu"]:
                    # Extract angular velocity and linear acceleration
                    av = msg.angular_velocity
                    la = msg.linear_acceleration
                    imu_rows.append([
                        timestamp,
                        av.x, av.y, av.z,
                        la.x, la.y, la.z
                    ])
        
        # Save CSVs
        def save_csv(path, header, rows):
            """Write rows to a CSV file with a header row."""
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in rows:
                    writer.writerow([repr(val) if isinstance(val, float) else val for val in row])


        save_csv(os.path.join(output_dir, f"{session_name}_eeg.csv"), ["timestamp"] + CHANNEL_NAMES, eeg_rows)
        save_csv(os.path.join(output_dir, f"{session_name}_gps.csv"), ["timestamp", "linear_x", "linear_y"], gps_rows)
        save_csv(os.path.join(output_dir, f"{session_name}_odom.csv"), ["timestamp", "yaw", "yaw_rate"], odom_rows)
        save_csv(os.path.join(output_dir, f"{session_name}_image_index.csv"), ["timestamp", "image_path"], image_rows)
        save_csv(os.path.join(output_dir, f"{session_name}_imu.csv"), ["timestamp", "ang_vel_x", "ang_vel_y", "ang_vel_z", "lin_acc_x", "lin_acc_y", "lin_acc_z"], imu_rows)


        elapsed = time.time() - start_time
        log_lines.append(f"Saved {total_samples} EEG | {len(gps_rows)} GPS | {len(odom_rows)} ODOM | {len(image_rows)} IMAGES | {len(imu_rows)} IMU")
        log_lines.append(f"Time: {elapsed:.2f} sec ({elapsed/60:.2f} min)")            


# Write log file
log_file = os.path.join(base_output_dir, "extraction_log.txt")
with open(log_file, "w") as f:
    f.write("\n".join(log_lines))


print("All bags processed. Log saved to extraction_log.txt")