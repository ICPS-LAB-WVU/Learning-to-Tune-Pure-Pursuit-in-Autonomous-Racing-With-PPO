# Learning-to-Tune-Pure-Pursuit-in-Autonomous-Racing: Joint-Lookahead-and-Steering-Gain-Control-With-PPO



This repository provides code, trained models, and example files related to the paper:

> **Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control With PPO**
> Mohamed Elgouhary and Amr S. El-Wakeel
> *IEEE Robotics and Automation Letters*, vol. 11, no. 4, pp. 5041–5048, 2026.
> DOI: [10.1109/LRA.2026.3669765](https://doi.org/10.1109/LRA.2026.3669765)

## Overview

Pure Pursuit is widely used in autonomous racing because it is simple, interpretable, and efficient enough for real-time control. However, its performance depends strongly on controller parameters such as the lookahead distance and steering gain.

This project uses **Proximal Policy Optimization (PPO)** to tune Pure Pursuit online. Instead of replacing the classical Pure Pursuit controller with an end-to-end neural controller, the learned policy adjusts two interpretable controller parameters:

* Lookahead distance, `Ld`
* Steering gain, `g`

The goal is to improve path tracking, lap time, steering smoothness, and generalization while preserving the practical advantages of a classical geometric controller.

## Key Idea

The controller keeps the Pure Pursuit structure but uses a learned PPO policy to select the lookahead distance and steering gain at each control step.

The policy observes compact driving features such as:

* Vehicle speed
* Near-horizon curvature
* Mid-horizon curvature
* Far-horizon curvature

The policy outputs:

```text
[Ld, g]
```

where:

* `Ld` controls how far ahead the Pure Pursuit controller targets a waypoint.
* `g` scales the steering response.

This design makes the method easier to interpret and deploy compared with fully end-to-end reinforcement-learning controllers.

## Repository Contents

```text
Learning-to-Tune-Pure-Pursuit-in-Autonomous-Racing-With-PPO/
├── README.md
└── src/
    ├── csv_data/
    ├── iv_pkg/
    ├── output/
    ├── YasMarina.csv
    ├── YasMarina.png
    ├── ppo_lookahead_gain_model.zip
    ├── ppo_lookahead_gain_model2.zip
    ├── vecnorm.pkl
    └── vecnorm2.pkl
```

## Main Components

| Path                                | Description                                                              |
| ----------------------------------- | ------------------------------------------------------------------------ |
| `src/iv_pkg/`                       | ROS 2 Python package containing the Pure Pursuit evaluation/control code |
| `src/iv_pkg/pure_pursuit_eval.py`   | Main evaluation/control script                                           |
| `src/csv_data/`                     | Track/raceline CSV files                                                 |
| `src/output/`                       | Output or generated experiment files                                     |
| `src/YasMarina.csv`                 | Example track/raceline file                                              |
| `src/YasMarina.png`                 | Track image/visualization                                                |
| `src/ppo_lookahead_gain_model.zip`  | Trained PPO policy for lookahead and steering-gain tuning                |
| `src/ppo_lookahead_gain_model2.zip` | Additional trained PPO policy checkpoint                                 |
| `src/vecnorm.pkl`                   | Normalization statistics used with the trained policy                    |
| `src/vecnorm2.pkl`                  | Additional normalization statistics                                      |

## Method Summary

The proposed RL–Pure Pursuit framework works as follows:

1. The vehicle follows a global raceline using a Pure Pursuit controller.
2. At each control step, the system extracts compact state features from the vehicle state and upcoming raceline geometry.
3. A PPO policy predicts the lookahead distance and steering gain.
4. These learned parameters are passed to the Pure Pursuit controller.
5. The Pure Pursuit controller computes the steering command.
6. The controller is evaluated in autonomous racing scenarios using simulation and real-car deployment.

## Why This Repository Is Useful

This repository can help researchers and students who are interested in:

* Autonomous racing
* F1TENTH control
* ROS 2 vehicle control
* Pure Pursuit path tracking
* Reinforcement learning for controller tuning
* Sim-to-real autonomous driving
* PPO-based parameter adaptation
* Learning-augmented classical control

## Installation

### 1. Clone the repository

Clone this repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ICPS-LAB-WVU/Learning-to-Tune-Pure-Pursuit-in-Autonomous-Racing-With-PPO.git
```

### 2. Build the workspace

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 3. Install Python dependencies

The exact environment may depend on your ROS 2 and F1TENTH setup. The main Python dependencies include:

```bash
pip install numpy scipy pandas matplotlib stable-baselines3 gym
```

For ROS 2, make sure the following packages are available:

* `rclpy`
* `nav_msgs`
* `geometry_msgs`
* `sensor_msgs`
* `ackermann_msgs`
* `visualization_msgs`

## Running the Code

After building and sourcing your workspace, run the main node/script from the ROS 2 package.

Example:

```bash
ros2 run iv_pkg pure_pursuit_eval
```

If your local package or executable name differs, check:

```bash
ros2 pkg list | grep iv_pkg
ros2 pkg executables iv_pkg
```

Then run the executable shown by ROS 2.

## Using the Trained PPO Policy

The repository includes trained PPO model files:

```text
src/ppo_lookahead_gain_model.zip
src/ppo_lookahead_gain_model2.zip
```

and normalization files:

```text
src/vecnorm.pkl
src/vecnorm2.pkl
```

When using the trained policy, make sure the model file and normalization file match the configuration used during training or evaluation.

A typical workflow is:

1. Load the raceline CSV.
2. Load the trained PPO model.
3. Load the corresponding normalization statistics.
4. Compute vehicle state and curvature features.
5. Predict `[Ld, g]`.
6. Apply the predicted parameters to the Pure Pursuit controller.
7. Publish the steering command to the vehicle or simulator.

## Track and Raceline Files

The repository includes an example Yas Marina track/raceline file:

```text
src/YasMarina.csv
```

and a track image:

```text
src/YasMarina.png
```

To use another track, prepare a compatible raceline CSV file and update the file path in the code or launch configuration.

## Citation

If this repository or the associated method helps your research, please cite the paper:

```bibtex
@article{elgouhary2026learning,
  author={Elgouhary, Mohamed and El-Wakeel, Amr S.},
  title={Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control With PPO},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={4},
  pages={5041--5048},
  year={2026},
  doi={10.1109/LRA.2026.3669765}
}
```

## Paper Links

* IEEE DOI: https://doi.org/10.1109/LRA.2026.3669765
* arXiv: https://arxiv.org/abs/2602.18386

## Recommended Citation Text

You may cite this work as:

> M. Elgouhary and A. S. El-Wakeel, “Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control With PPO,” *IEEE Robotics and Automation Letters*, vol. 11, no. 4, pp. 5041–5048, 2026.

## How to Reuse This Work

You can use this repository to:

* Compare PPO-tuned Pure Pursuit against fixed-lookahead Pure Pursuit.
* Study how lookahead distance affects autonomous racing performance.
* Study how steering-gain tuning affects tracking accuracy and smoothness.
* Build learning-augmented classical controllers.
* Extend the policy input space with additional vehicle or perception features.
* Test the method on new F1TENTH tracks.
* Adapt the learned tuning idea to other path-tracking controllers.

## Suggested Experiments

Researchers can extend this work by testing:

1. New tracks and unseen racelines.
2. Different PPO observation spaces.
3. Different reward functions.
4. Different speed profiles.
5. Comparison with Stanley, MPC, MPPI, or other controllers.
6. Sim-to-real transfer on additional F1TENTH platforms.
7. Safety filters or runtime monitors around the learned parameter outputs.

## Notes for Reproducibility

To reproduce results as closely as possible:

* Use the same trained PPO model and normalization file.
* Use the same raceline format.
* Use the same vehicle parameters.
* Verify the ROS 2 topic names before running.
* Confirm that the Ackermann drive topic matches your simulator or real vehicle.
* Test in simulation before deploying on hardware.


## Authors

**Mohamed Elgouhary**
Lane Department of Computer Science and Electrical Engineering
West Virginia University

**Amr S. El-Wakeel**
Lane Department of Computer Science and Electrical Engineering
West Virginia University


## Acknowledgment

This repository is associated with research conducted at the iCPS Lab, West Virginia University.

## Contact

For questions about the paper or repository, please contact:

**Mohamed Elgouhary**
Email: [mae00018@mix.wvu.edu](mailto:mae00018@mix.wvu.edu)
