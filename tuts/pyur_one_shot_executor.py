#!/usr/bin/env python3
"""Run one remote velocity pulse against the PyUR simulation."""

import math
from queue import Empty, Queue
import time

import rospy
import tf2_ros
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

POSITION_CONTROLLER = "scaled_pos_joint_traj_controller"
VELOCITY_CONTROLLER = "joint_group_vel_controller"
SWITCH_SERVICE = "/robot/controller_manager/switch_controller"
VELOCITY_TOPIC = "/joint_group_vel_controller/command"
JOINTS_PARAM = f"/robot/{VELOCITY_CONTROLLER}/joints"
PULSE_SECONDS = 3.0
MAX_SPEED = 0.05


def switch_controller(service, start, stop):
    request = SwitchControllerRequest()
    request.start_controllers = [start]
    request.stop_controllers = [stop]
    request.strictness = SwitchControllerRequest.STRICT
    request.timeout = 2.0
    if not service(request).ok:
        raise RuntimeError(f"Could not switch from {stop} to {start}")


def wait_for_subscriber(publisher):
    deadline = time.monotonic() + 5.0
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if publisher.get_num_connections():
            return
        time.sleep(0.02)
    raise RuntimeError(f"No subscriber connected to {publisher.resolved_name}")


def tcp_position(buffer, base, tool):
    transform = buffer.lookup_transform(base, tool, rospy.Time(0), rospy.Duration(3))
    position = transform.transform.translation
    return (position.x, position.y, position.z)


def main():
    rospy.init_node("pyur_one_shot_executor")
    try:
        rospy.wait_for_service(SWITCH_SERVICE, timeout=5)
    except rospy.ROSException as error:
        raise RuntimeError(
            "PyUR's controller manager is unavailable on the current ROS master. "
            "Source ros_talker_ipconfig.sh in both terminals and start "
            "'roslaunch bullet_ros_ur simulation.launch' before rerunning the executor."
        ) from error
    joint_names = rospy.get_param(JOINTS_PARAM, [])
    if not joint_names:
        raise RuntimeError(f"Missing velocity controller joint configuration at {JOINTS_PARAM}")
    base = rospy.get_param("~base_frame", "base_link")
    tool = rospy.get_param("~tool_frame", "tool0")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    observations = rospy.Publisher(
        "/mpc/test_observation", JointState, queue_size=1, latch=True
    )
    velocity = rospy.Publisher(VELOCITY_TOPIC, Float64MultiArray, latch=True, queue_size=1)
    observation = rospy.wait_for_message("/joint_states", JointState, timeout=5)
    if observation.header.stamp == rospy.Time(0):
        raise ValueError("PyUR returned an unstamped joint observation")
    replies = Queue()
    command_subscriber = rospy.Subscriber(
        "/mpc/test_command", JointTrajectory, replies.put, queue_size=1
    )
    try:
        deadline = time.monotonic() + 15
        saw_observation_subscriber = False
        rospy.loginfo("Sending one joint sample; retrying delivery until a matching reply arrives.")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            observations.publish(observation)
            saw_observation_subscriber |= observations.get_num_connections() > 0
            try:
                reply = replies.get(timeout=0.5)
            except Empty:
                continue
            if reply.header.stamp == observation.header.stamp:
                break
        else:
            raise RuntimeError(
                "No matching command on /mpc/test_command. Observation TCPROS "
                f"subscriber connected during this run: {saw_observation_subscriber}."
            )
    finally:
        command_subscriber.unregister()
        observations.unregister()

    if len(reply.points) != 1:
        raise ValueError("Expected one command point")
    speeds = reply.points[0].velocities
    if len(speeds) != len(joint_names) or sorted(reply.joint_names) != sorted(joint_names):
        raise ValueError("Expected one velocity for each configured arm joint")
    if any(not math.isfinite(speed) or abs(speed) > MAX_SPEED for speed in speeds):
        raise ValueError(f"Joint velocities must be finite and within ±{MAX_SPEED} rad/s")
    by_name = dict(zip(reply.joint_names, speeds))
    command = Float64MultiArray(data=[by_name[name] for name in joint_names])
    rospy.loginfo("Accepted joint velocities in rad/s: %s", by_name)

    switch = rospy.ServiceProxy(SWITCH_SERVICE, SwitchController)
    before = tcp_position(tf_buffer, base, tool)
    switch_controller(switch, VELOCITY_CONTROLLER, POSITION_CONTROLLER)
    try:
        wait_for_subscriber(velocity)
        velocity.publish(command)
        deadline = time.monotonic() + PULSE_SECONDS
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        try:
            velocity.publish(Float64MultiArray(data=[0.0] * len(joint_names)))
            time.sleep(0.1)
        finally:
            switch_controller(switch, POSITION_CONTROLLER, VELOCITY_CONTROLLER)

    time.sleep(0.2)
    after = tcp_position(tf_buffer, base, tool)
    delta = tuple(end - start for start, end in zip(before, after))
    rospy.loginfo("%s position in %s before [m]: %s", tool, base, before)
    rospy.loginfo("%s position in %s after  [m]: %s", tool, base, after)
    rospy.loginfo("Displacement XYZ [m]: %s; distance [mm]: %.3f", delta, 1000 * math.dist(before, after))


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, tf2_ros.TransformException, RuntimeError, KeyError, ValueError) as error:
        rospy.logerr("%s", error)
        raise SystemExit(1)
