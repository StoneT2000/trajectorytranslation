import sapien

sapien_v2 = True

if not hasattr(sapien, "__version__"):
    sapien_v2 = False

import tr2.envs.xmagical.traj_env

if sapien_v2:
    import tr2.envs.blockstacking
    import tr2.envs.boxpusher.env
    import tr2.envs.boxpusher.traj_env
    import tr2.envs.couchmoving.traj_env
else:
    import tr2.envs.opendrawer.traj_env
