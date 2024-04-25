# Autodrone for Person Re-Identification
![Our Auto-Drone](figures/drone.png)

## Background
For our autodrone project, we address challenges in hazardous environments, aiding first responders by providing imagery inside buildings and detecting people using autodrone. Current solutions involve line or site sensors, video/thermal feeds, and varying levels of automated detection or human monitoring, targeting private industry, first responders, and governments. Our proposed solution includes drone hardware featuring a 360-degree rotational camera and a 4-DOF robot arm. We integrate object detection using YOLOv9 on Gazebo Simulation, deploy on Jetson Orin Nano with TensorRT acceleration, and enable 3D reconstruction through camera-LiDAR fusion. Communication is facilitated by Quectel 5G Modem with OpenAir 5G SA Network. Our solution’s performance includes successful integration, data collection, 3D point cloud rendering, and object detection validation. The next steps involve obstacle avoidance, advanced algorithms, PCB design, motion control enhancement, and real-world tests with YOLOv9 on Jetson Orin Nano.

## Methods
- **Focus 1:** Object Detection and Application-[Object detection](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/blob/main/ObjectDetection); [Gazebo simulation](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/blob/main/Gazebo_simulation)
<p align="center">
  <img src="figures/gazebo_simulation.gif" alt="recon" width="600" height="335.25">
</p>
<p align="center">Gazebo Simulation of Object Detection of Drone using YOLOv9</p>

- **Focus 2:** 3D Reconstruction-[Gaussian Splatting](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/blob/main/Reconstruction/3dgs_depth/README.md); [Reconstruct with Lidar](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/main/Reconstruction)
<p align="center">
  <img src="figures/3dgs_ryon.gif" alt="recon" width="600" height="335.25">
</p>
<p align="center">3D Gaussian Splatting representation of Ryon Lab, visualized with <a href="https://github.com/buaacyw/GaussianEditor">GaussianEditor GUI</a>.</p>

- **Focus 3:** Wireless
- **Focus 4:** Hardware

## Report
Autodrone team final report from ELEC 594 Capstone Project in Spring 2024 at Rice University：
[ELEC594 Autodrone_project_final_report](https://github.com/Rice-MECE-Capstone-Projects/Autodrone/blob/main/Report/ELEC594_Autodrone_project_final_report.pdf)


This project is licensed under Electrical and Computer Engineering Department at Rice University

<img src="https://riceconnect.rice.edu/image/engineering/ece/SOE-ECE-Rice-logo-stacked.jpg" width="500" height="140" />
