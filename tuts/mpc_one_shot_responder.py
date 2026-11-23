#!/usr/bin/env python3
"""Reply to one PyUR observation with a fixed test velocity."""

import time

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

SHOULDER_SPEED = 0.05  # rad/s

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    rospy.init_node("mpc_one_shot_responder")
    commands = rospy.Publisher("/mpc/test_command", JointTrajectory, queue_size=1, latch=True)
    rospy.loginfo("Waiting for a PyUR joint observation")
    observation = rospy.wait_for_message("/mpc/test_observation", JointState)
    positions = dict(zip(observation.name, observation.position))
    velocities = dict(zip(observation.name, observation.velocity))
    rospy.loginfo("Observed positions [rad]: %s", {name: positions[name] for name in ARM_JOINTS})
    rospy.loginfo("Observed velocities [rad/s]: %s", {name: velocities[name] for name in ARM_JOINTS})

    deadline = time.monotonic() + 5.0
    while not commands.get_num_connections():
        if rospy.is_shutdown() or time.monotonic() >= deadline:
            raise RuntimeError("Executor did not subscribe to the command topic")
        time.sleep(0.02)

    reply = JointTrajectory()
    reply.header.stamp = observation.header.stamp
    reply.joint_names = ARM_JOINTS
    reply.points = [JointTrajectoryPoint(velocities=[SHOULDER_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0])]
    commands.publish(reply)
    rospy.loginfo("Sent one shoulder_pan_joint velocity command at %.2f rad/s", SHOULDER_SPEED)
    while not rospy.is_shutdown() and commands.get_num_connections() and time.monotonic() < deadline:
        time.sleep(0.02)


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, KeyError, RuntimeError) as error:
        rospy.logerr("%s", error)
        raise SystemExit(1)
