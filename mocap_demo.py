from roboschool import cpp_household, scene_abstract

import gym, os
import numpy as np
from resl_test.gym_mocap_walker import RoboschoolMocapHumanoid
from asf_parser import AsfParser, sanitize


def demo_run():
    env = gym.make("RoboschoolMocapHumanoid-v1")

    parser = AsfParser()
    parser.parse('examples/12.asf')
    parser.save_json('humanoid_mocap.json')
    parser.save_mujoco_xml('humanoid_mocap.xml')

    # parse AMC file (animation)
    amc_file = open('examples/02_01.amc', 'r')
    frames = []  # list of dicts {property: value(s)}
    current_frame = {}
    for i, line in enumerate(amc_file):
        if i < 3:
            continue
        if line.startswith('root'):
            if len(current_frame) > 0:
                frames.append(current_frame)
            current_frame = {}
        split = line.split(' ')
        if len(split) <= 1:
            continue
        current_frame[split[0]] = np.array([float(x) for x in split[1:]])
    if len(current_frame) > 0:
        frames.append(current_frame)

    frame_nr = 1
    env.reset()

    # disable physics for this scene
    gravity = 9.8  # has no effect
    env.env.scene.cpp_world = cpp_household.World(gravity, 0)
    env.env.scene.cpp_world.set_glsl_path(os.path.join(os.path.dirname(scene_abstract.__file__), "cpp-household/glsl"))
    env.reset()

    joints = env.env.motor_names
    old_pos = frames[0]['root'][:3]
    while 1:
        frame = 0
        score = 0

        obs, r, done, _ = env.step(np.zeros(56))
        if frame % 50 == 0:
            pos = frames[frame_nr]['root'][:3]
            delta_pos = sanitize((pos - old_pos) / 16.)
            env.env.move_robot(delta_pos[0], delta_pos[1], delta_pos[2])

            for joint in joints:
                if 'back' in joint:
                    continue
                value = 0
                if '_' in joint:
                    joint_name = joint[:-2]
                    if len(frames[frame_nr][joint_name]) == 1:
                        value = frames[frame_nr][joint_name][0]
                    elif joint.endswith('z'):
                        value = frames[frame_nr][joint_name][-1]
                    elif joint.endswith('x'):
                        value = frames[frame_nr][joint_name][0]
                    elif joint.endswith('y'):
                        if joint_name+'_x' in joints:
                            value = -frames[frame_nr][joint_name][1]
                        else:
                            value = -frames[frame_nr][joint_name][0]

                else:
                    # case does not occur
                    value = frames[frame_nr][joint]
                env.env.jdict[joint].reset_current_position(value * np.pi / 180., 0)

            frame_nr += 1
            old_pos = pos
            if frame_nr >= len(frames):
                print('RESET ENVIRONMENT')
                frame_nr = 0
                env.reset()
                for joint in joints:
                    env.env.jdict[joint].reset_current_position(0, 0)
                old_pos = frames[frame_nr]['root'][:3]

        score += r
        frame += 1
        still_open = env.render("human")
        if not still_open:
            return
            # if not done: continue
            # if restart_delay==0:
            #     print("score=%0.2f in %i frames" % (score, frame))
            #     if still_open!=True:      # not True in multiplayer or non-Roboschool environment
            #         break
            #     restart_delay = 60*2  # 2 sec at 60 fps
            # restart_delay -= 1
            # if restart_delay==0: break


if __name__ == "__main__":
    demo_run()
