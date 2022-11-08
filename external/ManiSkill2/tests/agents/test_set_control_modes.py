import time

from mani_skill2.envs.pick_and_place.lift_cube import LiftCubeFixedXmate3RobotiqEnv


def main():
    env = LiftCubeFixedXmate3RobotiqEnv()
    env.reset()
    viewer = env.render()
    supported_control_modes = env.agent.supported_control_modes
    print("supported control modes: ", supported_control_modes)
    print(
        "action_space: ",
    )
    for k, v in env.action_space.items():
        print(k, v.shape)
    num_supported_control_modes = len(supported_control_modes)
    mode_idx = 0
    control_mode = supported_control_modes[mode_idx]
    while not viewer.closed:
        if viewer.window.key_down("r"):
            env.reset()
            viewer = env.render()
        if viewer.window.key_down("c"):
            time.sleep(0.5)
            mode_idx += 1
            if mode_idx >= num_supported_control_modes:
                print("All control modes tested. Exit.")
                exit()
            control_mode = supported_control_modes[mode_idx]
            print(f"#### Set control mode to [{control_mode}] ####")
        env.render()
        action = {
            "control_mode": control_mode,
            "action": env.action_space[control_mode].low,
        }
        obs, rew, done, info = env.step(action)
        # print(rew, done, info)


if __name__ == "__main__":
    main()
