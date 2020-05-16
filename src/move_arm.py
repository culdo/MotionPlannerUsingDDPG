from core.manipulator import MoveGroup
import rospy

def main():
    try:
        # begin the tutorial by setting up the moveit_commander (press ctrl-d to exit)
        tutorial = MoveGroup()

        # execute a movement using a joint state goal
        tutorial.go_to_joint_state([0.0, 0.8, -0.5, 0.8])
        tutorial.go_to_joint_state([0.0, -1.0, 0.314, 0.707])

        # execute a movement using a pose goal
        # tutorial.go_to_pose_goal()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()