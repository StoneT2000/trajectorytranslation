# block support is only this region!
# random_xy = self._episode_rng.uniform([-0.02,-0.07],[0.1,0.07])

tower_configs = {

}
for i in range(3, 10):
    tower_configs[i] = dict(
        generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.14 + rng.rand()*0.05],
        offset=([0.06,0.04],[0.09, 0.06])
    )


pyramid_configs = {
    5: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.01],[0.08, 0.03])),
    4: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.02],[0.08, 0.04])),
    3: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.02],[0.08, 0.06])),
    2: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.02],[0.08, 0.06]))
}

minecraft_villagerhouse_configs = {
    9: dict(generate_spawn_location=lambda rng: [-0.03, -0.11],
    offset=([0.05,0.0],[0.08, 0.04])),
    25: dict(generate_spawn_location=lambda rng: [-0.04, 0],
    offset=([0.07,-0.07],[0.07, 0.001])),
}

minecraft_creeper_configs = {
    10:  dict(generate_spawn_location=lambda rng: [-0.03, -0.11],
    offset=([0.07,0.0],[0.08, 0.0])),
}
# 0.420601715, 0.003660263, 0.021030488

# To calibrate, return xarm out of view, declare a (0, -0.1) point.
# use pose estimation and set negatvi
ROBOT_Y_OFFSET = -0.002001 - 0.1
ROBOT_X_OFFSET = -0.43729989
realtower = dict()
start_x = 0.492435663
start_y=0.035661391

# start_x = 0.471230956
for i in range(2, 10):
    realtower[i]= dict(generate_spawn_location=lambda rng: [start_x + ROBOT_X_OFFSET, start_y + ROBOT_Y_OFFSET], offset=([0.0,0.06],[0.0, 0.06]))
realtower[8]['offset'] = ([0.076, 0.049], [0.076, 0.049])

realtower2 = dict()
for i in range(2, 10):
    realtower2[i]= dict(generate_spawn_location=lambda rng: [start_x + ROBOT_X_OFFSET, start_y + ROBOT_Y_OFFSET], offset=([0.0,0.02],[0.0, 0.02]))

custom_mc_scene_1 = dict()
custom_mc_scene_1 = {
    5: dict(
        generate_spawn_location=lambda rng: [-0.01 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02], offset=([0.04,0.07],[0.04, 0.07])
    )
}

realcustom_mc_scene_1 = {
    5: dict(
        generate_spawn_location=lambda rng: [-0.01 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02], offset=([0.04,0.07],[0.04, 0.07])
    )
}

realcustom_mc_scene_2 = {
    8: dict(
        generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.01, -0.12 + rng.rand()*0.0], offset=([0.06,0.09],[0.06, 0.09])
    )
}
realcustom_mc_scene_3 = {
    7: dict(
        generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.01, -0.12 + rng.rand()*0.0], offset=([0.05,0.065],[0.05, 0.065])
    )
}

realcustom_mc_scene_4 = {
    # mini castle - release height 0.025
    28: dict(
        generate_spawn_location=lambda rng: [-0.06 + rng.rand() * 0.01, -0.14 + rng.rand()*0.0], offset=([0.05,0.055],[0.05, 0.055])
    ),
}

# cubes = {
#     9: dict(
#         generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.01, -0.12 + rng.rand()*0.0], offset=([0.02,0.09],[0.02, 0.09])
#     ),
#     27: dict(
#         generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.01, -0.14 + rng.rand()*0.0], offset=([0.03,0.06],[0.03, 0.06])
#     ),
    
#     36: dict(
#         generate_spawn_location=lambda rng: [-0.04 + rng.rand() * 0.01, -0.14 + rng.rand()*0.0], offset=([0.045,0.055],[0.045, 0.055])
#     )
# }



realpyramid_configs = {
    5: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.01],[0.08, 0.03])),
    4: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.04,0.04],[0.04, 0.04])),
    3: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.08,0.05],[0.08, 0.05])),
    2: dict(generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.02, -0.12 + rng.rand()*0.02],
    offset=([0.06,-0.02],[0.08, 0.06]))
}

# for i in range(3, 10):
#     tower_configs[i] = dict(
#         generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.1, -0.14 + rng.rand()*0.04],
#         offset=([0.0,0.06],[0.01, 0.06])
#     )





### LARGE MODEL CONFIGS


# large model
# for i in range(2, 10):
#     realtower[i]= dict(generate_spawn_location=lambda rng: [start_x + ROBOT_X_OFFSET, start_y + ROBOT_Y_OFFSET], offset=([-0.04,0.1],[-0.04, 0.1]))
# for i in range(3, 10):
#     tower_configs[i] = dict(
#         generate_spawn_location=lambda rng: [-0.02 + rng.rand() * 0.05, -0.15 + rng.rand()*0.05],
#         offset=([-0.06,0.12],[-0.09, 0.12])
#     )