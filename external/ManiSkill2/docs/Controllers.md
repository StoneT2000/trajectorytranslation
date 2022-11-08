# Combined Controller

The agent (robot) is actuated by the `combined_controller`. Since the robot is composed of several components (arm,
gripper, mobile platform, ...), we use separate controllers to control each component. The `combined_controller` will
separate the input `action` to the `action` of each controller.

## Control mode
An `agent` supports multiple control mode. Each control mode corresponds to one `combined_controller`. All `combined_comtroller` will be initialized at the initialization of the `agent`. User can switch control mode in the middle of one task.

Demo:
```bash
python tests/agents/test_set_control_modes.py
```

# Controllers

- [General Controllers](#general-controllers)
    - [General PD Joint Position](#general-pd-joint-position) (general_pd_joint_pos.py)
    - [General PD Joint Velocity](#general-pd-joint-velocity) (general_pd_joint_vel.py)
    - [General PD Joint Position and Velocity](#general-pd-joint-position-and-velocity) (general_pd_joint_pos_vel.py)
- [Arm Controllers](#arm-controllers)
    - [Arm Impedance End Effector Pose](#arm-impedance-end-effector-pose) (arm_imp_ee_pos.py)
    - [Arm Impedance Joint Position](#arm-impedance-joint-position) (arm_imp_joint_pos.py)
    - [Arm PD End Effector Delta Position](#arm-pd-end-effector-delta-position) (arm_pd_ee_delta_position.py)
- [Gripper Controllers](#gripper-controllers)
    - [Gripper PD Joint Position Mimic](#gripper-pd-joint-position-mimic) (gripper_pd_joint_pos_mimic.py)
    - [Gripper PD Joint Velocity Mimic](#gripper-pd-joint-velocity-mimic) (gripper_pd_joint_vel_mimic.py)
- [Mobile Platform Controllers](#mobile-platform-controllers)
    - [Mobile PD Joint Velocity Decoupled](#mobile-pd-joint-velocity-decoupled) (mobile_pd_joint_vel_decoupled.py)
    - [Mobile PD Joint Velocity Differential Drive](#mobile-pd-joint-velocity-differential-drive) (
      mobile_pd_joint_vel_diff.py)

## General Controllers

### General PD Joint Position

### General PD Joint Velocity

### General PD Joint Position and Velocity

## Arm Controllers

### Arm Impedance End Effector Pose

### Arm Impedance Joint Position

### Arm PD End Effector Delta Position

## Gripper Controllers

### Gripper PD Joint Position Mimic

### Gripper PD Joint Velocity Mimic

## Mobile Platform Controllers

### Mobile PD Joint Velocity Decoupled

### Mobile PD Joint Velocity Differential Drive

[Differential Drive](http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf)
