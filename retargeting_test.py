import argparse, math, rospy
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header, Bool, String
from tf.transformations import quaternion_about_axis, quaternion_multiply

FRAME_ID   = "tracker_base"
PUB_TOPIC  = "tracker_pose"
PUB_HZ     = 130.0

SPEED      = 0.1
SIDE_LEN   = 0.10
PERIM      = 4.0 * SIDE_LEN

ANGLE_DEG  = 30.0
ORI_PERIOD = 4.0

TRIGGER_TEXT = "Upperbody Mode is Changed to #10 (3D Mouse Mode)"

READY_POS = [
    [-1.9777027368545532, -0.10151958465576172, 0.45683467388153076],
    [-1.7924025058746338,  0.037916362285614014, 0.8278588056564331],
    [-1.6809483766555786, -0.27021169662475586, 0.78521728515625],
    [-1.3718138933181763,  0.21830034255981445, 0.706558883190155],
    [-1.6898536682128906,  0.2616920471191406, 0.7718719840049744],
    [-1.379732370376587 , -0.2619709372520447 , 0.6905632615089417],
    [-0.3161889314651489, -0.05778980255126953, -0.007666349411010742]
]

READY_ORI = [
    [ 0.032963041216135025,  0.9978668689727783 , -0.005082550458610058, -0.056119345128536224],
    [-0.1073051169514656  ,  0.09736108779907227,  0.9826999306678772 ,  0.11535760015249252 ],
    [ 0.24631604552268982 , -0.40684595704078674, -0.6926330327987671 ,  0.5422769784927368  ],
    [-0.5134174823760986  , -0.39466747641563416,  0.6021975874900818 ,  0.4669029116630554  ],
    [ 0.1449253112077713  ,  0.08316605538129807,  0.739422619342804  ,  0.6521766781806946  ],
    [-0.48006588220596313 ,  0.5306476950645447 ,  0.5378783941268921 , -0.44568687677383423],
    [ 0.005975766107439995, -0.07444379478693008, -0.07127542793750763,  0.9946569204330444  ]
]

class RightHandSquarePub:
    def __init__(self, mode):
        self.mode     = mode.lower()       # xy/xz/yz/roll/pitch/yaw
        self.enabled  = False
        self.pub      = rospy.Publisher(PUB_TOPIC, PoseArray, queue_size=10)
        self.status_p = rospy.Publisher("/TRACKERSTATUS", Bool, queue_size=10)
        rospy.Subscriber("/tocabi/guilog", String, self.guilog_cb)

        self.static_poses = []
        for i in [0,1,2,3,4,5,6]:
            p = Pose()
            p.position.x, p.position.y, p.position.z = READY_POS[i]
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = READY_ORI[i]
            self.static_poses.append(p)

        self.rh_x0, self.rh_y0, self.rh_z0 = READY_POS[5]
        self.rh_q = READY_ORI[5]

        self.start_time = None

        self.ready_pose = PoseArray()
        self.ready_pose.poses.extend(self.static_poses[0:1])  # pelvis
        self.ready_pose.poses.extend(self.static_poses[1:2])  # chest
        self.ready_pose.poses.extend(self.static_poses[2:3])  # Lupperarm
        self.ready_pose.poses.extend(self.static_poses[3:4])  # Lhand
        self.ready_pose.poses.extend(self.static_poses[4:5])  # Rupperarm
        self.ready_pose.poses.extend(self.static_poses[5:6])  # Rhand
        self.ready_pose.poses.extend(self.static_poses[6:7])  # HMD

    def guilog_cb(self, msg):
        if TRIGGER_TEXT in msg.data and not self.enabled:
            rospy.loginfo("Trigger received publishing enabled")
            self.enabled    = True
            self.start_time = rospy.Time.now()

    def spin(self):
        rate = rospy.Rate(PUB_HZ)
        while not rospy.is_shutdown():
            if self.enabled:
                self.publish_tracker_pose()
            else:
                self.ready_pose.header = Header(stamp=rospy.Time.now(), frame_id=FRAME_ID)
                self.pub.publish(self.ready_pose)
                self.status_p.publish(Bool(data=True))

            rate.sleep()

    def publish_tracker_pose(self):
        now = rospy.Time.now()
        t   = (now - self.start_time).to_sec()

        x, y, z = self.rh_x0, self.rh_y0, self.rh_z0
        q       = self.rh_q[:]

        if self.mode in ("xy","xz","yz"):
            s = (t * SPEED) % PERIM
            if   s < SIDE_LEN:                u,v = 0.0, s
            elif s < 2*SIDE_LEN:              u,v = s - SIDE_LEN, SIDE_LEN
            elif s < 3*SIDE_LEN:              u,v = SIDE_LEN, SIDE_LEN - (s - 2*SIDE_LEN)
            else:                             u,v = SIDE_LEN - (s - 3*SIDE_LEN), 0.0

            if self.mode == "xy":   x += u; y += v
            elif self.mode == "xz": x += u; z += v
            else:                   y += u; z += v
        else:
            angle = math.radians(ANGLE_DEG) * math.sin(2*math.pi*t/ORI_PERIOD)
            axis  = {"roll":(1,0,0), "pitch":(0,1,0), "yaw":(0,0,1)}[self.mode]
            q_delta = quaternion_about_axis(angle, axis)
            q = quaternion_multiply(self.rh_q, q_delta)

        rh = Pose()
        rh.position.x, rh.position.y, rh.position.z = x, y, z
        rh.orientation.x, rh.orientation.y, rh.orientation.z, rh.orientation.w = q

        pa = PoseArray()
        pa.header = Header(stamp=now, frame_id=FRAME_ID)
        pa.poses.extend(self.static_poses[0:1])  # pelvis
        pa.poses.extend(self.static_poses[1:2])  # chest
        pa.poses.extend(self.static_poses[2:3])  # Lupperarm
        pa.poses.extend(self.static_poses[3:4])  # Lhand
        pa.poses.extend(self.static_poses[4:5])  # Rupperarm
        pa.poses.append(rh)                      # Rhand
        pa.poses.extend(self.static_poses[6:7])  # HMD

        self.pub.publish(pa)
        self.status_p.publish(Bool(data=True))

def main():
    parser = argparse.ArgumentParser(description="Righthand motion publisher (6modes)")
    parser.add_argument("--mode", choices=["xy","xz","yz","roll","pitch","yaw"], default="xy",
                        help="motion mode (default: xy)")
    args = parser.parse_args()

    rospy.init_node("right_hand_square_pub", anonymous=False)
    RightHandSquarePub(args.mode).spin()

if __name__ == "__main__":
    main()


# rosbag record /tocabi/robot_poses /tocabi/desired_robot_poses /tocabi/tracker_poses /tocabi/robot_joints /tocabi/desired_joints