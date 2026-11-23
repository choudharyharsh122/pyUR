#!/usr/bin/env python3
"""Run a bounded TCP integration test against the PyUR simulator (ROS Noetic)."""

import argparse
import json
import math
import socket
import time

import rosnode
import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray


POSITION = "scaled_pos_joint_traj_controller"
VELOCITY = "joint_group_vel_controller"
SWITCH = "/robot/controller_manager/switch_controller"
MAX_SPEED = 0.05
WATCHDOG = 0.75


def switch_controller(service, start, stop):
    request = SwitchControllerRequest(
        start_controllers=[start], stop_controllers=[stop],
        strictness=SwitchControllerRequest.STRICT, timeout=2.0,
    )
    if not service(request).ok:
        raise RuntimeError(f"Could not switch from {stop} to {start}")


def exchange(connection, latest_state, joints, publisher, approach_feedback=None, close_gripper=None, *,
             carry=False, open_gripper=None):
    last_command = [0.0] * len(joints)
    time_limit = 480.0 if open_gripper else 240.0 if carry else 120.0 if approach_feedback else 30.0
    deadline = time.monotonic() + time_limit
    connection.settimeout(WATCHDOG)
    closed = False
    with connection.makefile("rb") as stream:
        for seq in range(5200 if open_gripper else 2600 if carry else 1000):
            if rospy.is_shutdown() or time.monotonic() >= deadline:
                break
            state = latest_state()
            if state is None or time.monotonic() - state["received"] > 0.15:
                raise RuntimeError("No fresh PyUR arm feedback")
            observation = {"q": state["q"], "qd": state["qd"], "seq": seq,
                           "names": joints, "last_command": last_command}
            if approach_feedback:
                observation.update(approach_feedback())
            observation["gripper_closed"] = closed
            connection.sendall(json.dumps(observation, allow_nan=False).encode() + b"\n")
            line = stream.readline(4096)
            if not line:
                break
            if not line.endswith(b"\n"):
                raise ValueError("Invalid command framing")
            reply = json.loads(line)
            if reply.get("seq") != seq:
                raise ValueError("Command sequence does not match observation")
            if reply.get("stop") is True:
                break
            if time.monotonic() >= deadline or rospy.is_shutdown():
                break
            if reply.get("close_gripper") is True:
                if close_gripper is None or closed:
                    raise ValueError("Unexpected gripper close request")
                last_command = [0.0] * len(joints)
                publisher.publish(Float64MultiArray(data=last_command))
                if not close_gripper().success:
                    raise RuntimeError("PyUR rejected the RG6 close request")
                closed = True
                continue
            if reply.get("open_gripper") is True:
                if open_gripper is None or not closed:
                    raise ValueError("Unexpected gripper open request")
                if approach_feedback()["destination_load"] < 0.2:
                    raise RuntimeError("Refusing release without destination support")
                last_command = [0.0] * len(joints)
                publisher.publish(Float64MultiArray(data=last_command))
                if not open_gripper().success:
                    raise RuntimeError("PyUR rejected the RG6 open request")
                closed = False
                continue
            command = reply["velocity"]
            if (not isinstance(command, list) or len(command) != len(joints)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           or abs(v) > MAX_SPEED for v in command)):
                raise ValueError("Expected six finite velocities within ±0.05 rad/s")
            publisher.publish(Float64MultiArray(data=command))
            last_command = command


def make_approach_feedback(pick=False, place=False):
    import tf2_ros
    from bullet_ros_ur.msg import Collisions
    from geometry_msgs.msg import PoseArray
    from tf.transformations import quaternion_matrix
    from visualization_msgs.msg import MarkerArray

    buffer = tf2_ros.Buffer()
    listener = None if pick else tf2_ros.TransformListener(buffer)
    block = None
    contacts = None
    finger_poses = None

    def capture(message):
        nonlocal block
        for marker in message.markers:
            if marker.header.frame_id == "base_link" and marker.mesh_resource.endswith("/mission_block.obj"):
                p = marker.pose.position
                block = ([p.x, p.y, p.z], time.monotonic())
                return

    subscriber = rospy.Subscriber("/other_objects", MarkerArray, capture, queue_size=1)
    contact_subscriber = None
    finger_subscriber = None
    support_subscriber = None
    destination_load = None
    if place:
        def capture_support(message):
            nonlocal destination_load
            if math.isfinite(message.data):
                destination_load = (message.data, time.monotonic())

        support_subscriber = rospy.Subscriber("/bullet_ros/mission_destination_load", Float32,
                                             capture_support, queue_size=1)
    if pick:
        def capture_contacts(message):
            nonlocal contacts
            forces = {}
            for name, link, force in zip(message.obj_names, message.robot_links, message.forces):
                if name == "mission_block" and math.isfinite(force) and force > 0:
                    forces[link] = forces.get(link, 0.0) + force
            contacts = (forces, time.monotonic())

        contact_subscriber = rospy.Subscriber("/bullet_ros/collisions", Collisions,
                                              capture_contacts, queue_size=1)

        def capture_fingers(message):
            nonlocal finger_poses
            if message.header.frame_id == "base_link" and len(message.poses) == 2:
                finger_poses = (message.poses, time.monotonic())

        finger_subscriber = rospy.Subscriber("/bullet_ros/mission_fingers", PoseArray,
                                             capture_fingers, queue_size=1)

    def read(timeout=0.2):
        frames = ("left_inner_finger", "right_inner_finger")
        if pick:
            deadline = time.monotonic() + timeout
            while finger_poses is None and time.monotonic() < deadline and not rospy.is_shutdown():
                time.sleep(0.01)
            sample = finger_poses
            if sample is None or time.monotonic() - sample[1] > 0.25:
                raise RuntimeError("No fresh PyBullet RG6 finger poses. Relaunch the mission scene.")
            poses = [(pose.position, pose.orientation) for pose in sample[0]]
        else:
            latest = [buffer.lookup_transform("base_link", frame, rospy.Time(0), rospy.Duration(timeout))
                      for frame in frames]
            stamp = min(transform.header.stamp for transform in latest)
            age = (rospy.Time.now() - stamp).to_sec()
            if age > 0.5:
                raise RuntimeError(f"Stale RG6 finger transforms ({age:.3f} s old)")
            transforms = [buffer.lookup_transform("base_link", frame, stamp, rospy.Duration(0.2)).transform
                          for frame in frames]
            poses = [(transform.translation, transform.rotation) for transform in transforms]
        sample = block
        if sample is None or time.monotonic() - sample[1] > 0.25:
            raise RuntimeError("No fresh mission block marker. Launch ros_mission_approach.yaml.")
        fingers = []
        pads = []
        for p, r in poses:
            fingers.append((p.x, p.y, p.z))
            if pick:
                axis = quaternion_matrix((r.x, r.y, r.z, r.w))[:3, 2]
                # RG6 finger mesh extends about 67 mm along local +Z.
                pads.append([value + 0.045 * direction
                             for value, direction in zip(fingers[-1], axis)])
        opening = math.sqrt(sum((left - right) ** 2 for left, right in zip(*fingers)))
        if not pick and opening < 0.065:
            raise RuntimeError(f"Simulated RG6 finger-frame separation too small: {opening * 1000:.1f} mm")
        center = [(left + right) / 2 for left, right in zip(*fingers)]
        result = {"grasp_center_base_link": center, "block_center_base_link": sample[0]}
        if pick:
            contact_sample = contacts
            forces = (contact_sample[0] if contact_sample is not None
                      and time.monotonic() - contact_sample[1] <= 0.25 else {})
            per_jaw = [forces.get(frame, 0.0) for frame in frames]
            result.update(
                pad_center_base_link=[(left + right) / 2 for left, right in zip(*pads)],
                finger_separation=opening,
                block_contact_links=sorted(forces),
                per_jaw_force=per_jaw,
                bilateral_contact=min(per_jaw) >= 0.1,
            )
        if place:
            load = destination_load
            if load is None or time.monotonic() - load[1] > 0.25:
                raise RuntimeError("No fresh destination load. Restart the mission simulator.")
            result["destination_load"] = load[0]
        return result

    return read, (listener, subscriber, contact_subscriber, finger_subscriber, support_subscriber)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--grip-force", type=float, default=100.0,
                        help="Simulated RG6 close force setting for --pick and --carry")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--approach", action="store_true", help="Include RG6 and block feedback")
    mode.add_argument("--pick", action="store_true", help="Allow one RG6 close during PickLift")
    mode.add_argument("--carry", action="store_true", help="Allow a continuous PickLift and carry session")
    mode.add_argument("--place", action="store_true", help="Allow pick, carry, and a supported release")
    args = parser.parse_args()
    if not 0 < args.grip_force <= 180.0:
        parser.error("--grip-force must be greater than 0 and at most 180")

    rospy.init_node("pyur_mpc_test_bridge")
    if "/pyUR" not in rosnode.get_node_names():
        raise RuntimeError("Start the PyUR simulator on this ROS master first")
    rospy.wait_for_service(SWITCH, timeout=5)
    joints = rospy.get_param(f"/robot/{VELOCITY}/joints")
    if len(joints) != 6 or len(set(joints)) != 6:
        raise ValueError("Expected six distinct arm joints")
    state = None

    def capture(message):
        nonlocal state
        position = dict(zip(message.name, message.position))
        velocity = dict(zip(message.name, message.velocity))
        if all(name in position and name in velocity for name in joints):
            q = [position[name] for name in joints]
            qd = [velocity[name] for name in joints]
            if all(math.isfinite(value) for value in q + qd):
                state = {"q": q, "qd": qd, "received": time.monotonic()}

    feedback = rospy.Subscriber("/robot/joint_states", JointState, capture, queue_size=1)
    approach_feedback = None
    close_gripper = None
    open_gripper = None
    if args.approach or args.pick or args.carry or args.place:
        from bullet_ros_ur.srv import grip

        approach_feedback, approach_subscriptions = make_approach_feedback(not args.approach, args.place)
        grip_service = "/robot/ur_hardware_interface/grip"
        rospy.wait_for_service(grip_service, timeout=5)
        gripper = rospy.ServiceProxy(grip_service, grip)
        if args.approach or args.place:
            if not gripper(150.0, 100.0).success:
                raise RuntimeError("Could not open the simulated RG6 gripper")
        if not args.approach:
            def close_gripper():
                rospy.loginfo("Closing simulated RG6 with force setting %.1f", args.grip_force)
                return gripper(0.0, args.grip_force)
            if args.place:
                def open_gripper():
                    return gripper(150.0, 100.0)
    publisher = rospy.Publisher("/joint_group_vel_controller/command", Float64MultiArray,
                                queue_size=1)
    service = rospy.ServiceProxy(SWITCH, SwitchController)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        server.settimeout(1.0)
        rospy.loginfo("Waiting for MPC client on %s:%d", args.bind, args.port)
        while not rospy.is_shutdown():
            try:
                connection, _ = server.accept()
                break
            except socket.timeout:
                continue
        else:
            return

        with connection:
            if approach_feedback:
                approach_feedback(timeout=3.0)
            switch_controller(service, VELOCITY, POSITION)
            try:
                ready_by = time.monotonic() + 3.0
                while not publisher.get_num_connections() or state is None:
                    if rospy.is_shutdown() or time.monotonic() >= ready_by:
                        raise RuntimeError("PyUR joint feedback or velocity controller unavailable")
                    time.sleep(0.01)
                exchange(connection, lambda: state, joints, publisher, approach_feedback,
                         close_gripper, carry=args.carry or args.place, open_gripper=open_gripper)
            finally:
                try:
                    publisher.publish(Float64MultiArray(data=[0.0] * len(joints)))
                    time.sleep(0.1)
                finally:
                    switch_controller(service, POSITION, VELOCITY)
                    feedback.unregister()


if __name__ == "__main__":
    main()
