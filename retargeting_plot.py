#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tocabi rosbag  →  PoseArray + JointState 추출 & 시각화(Qt Tabs)
  • Δ(rhand) 3-D / 2-D Trajectories + Position Error
  • Right-arm 8-joint Robot vs Desired & Error
"""
#########################################################################
# 1) 기본 import
#########################################################################
import sys, os, numpy as np, rosbag
from typing import Dict, List

# ── GUI / Plot ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")        # 한 번만 지정
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTabWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


#########################################################################
# 2) rosbag → numpy 추출 (Δ 변환 포함)
#########################################################################
FILE_PATH = "10_gain/"
BAG_FILES = ["xy.bag", "xz.bag", "yz.bag"]
TOPIC_KEYS = {
    "/tocabi/robot_poses":         "robot_poses",
    "/tocabi/desired_robot_poses": "desired_robot_poses",
    "/tocabi/tracker_poses":       "tracker_poses",
    "/tocabi/robot_joints":        "robot_joints",
    "/tocabi/desired_joints":      "desired_joints",
}
LINK_ORDER = ["lhand", "head", "rhand"]
JOINT_NAMES = [  # 33개
    "L_HipYaw_Joint","L_HipRoll_Joint","L_HipPitch_Joint","L_Knee_Joint",
    "L_AnklePitch_Joint","L_AnkleRoll_Joint",
    "R_HipYaw_Joint","R_HipRoll_Joint","R_HipPitch_Joint","R_Knee_Joint",
    "R_AnklePitch_Joint","R_AnkleRoll_Joint",
    "Waist1_Joint","Waist2_Joint","Upperbody_Joint",
    "L_Shoulder1_Joint","L_Shoulder2_Joint","L_Shoulder3_Joint",
    "L_Armlink_Joint","L_Elbow_Joint","L_Forearm_Joint",
    "L_Wrist1_Joint","L_Wrist2_Joint","Neck_Joint","Head_Joint",
    "R_Shoulder1_Joint","R_Shoulder2_Joint","R_Shoulder3_Joint",
    "R_Armlink_Joint","R_Elbow_Joint","R_Forearm_Joint",
    "R_Wrist1_Joint","R_Wrist2_Joint",
]
JOINT_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def posearray_to_np(msg) -> np.ndarray:
    out = np.zeros((3, 7))
    for i, pose in enumerate(msg.poses[:3]):
        out[i, :3] = pose.position.x, pose.position.y, pose.position.z
        out[i, 3:] = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    return out


def jointstate_to_np(msg) -> np.ndarray:
    out = np.full(len(JOINT_NAMES), np.nan)
    name2idx = {n: i for i, n in enumerate(msg.name)}
    for j, name in enumerate(JOINT_NAMES):
        idx = name2idx.get(name)
        if idx is not None:
            out[j] = msg.position[idx]
    return out


def _delta_xyz(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[:, :, :3] -= arr[0:1, :, :3]
    return out


def extract_single_bag(path: str) -> Dict[str, np.ndarray]:
    buf: Dict[str, List[np.ndarray]] = {k: [] for k in TOPIC_KEYS.values()}
    with rosbag.Bag(path) as bag:
        for topic, msg, _ in bag.read_messages(topics=list(TOPIC_KEYS)):
            key = TOPIC_KEYS[topic]
            buf[key].append(
                posearray_to_np(msg) if topic.endswith("_poses")
                else jointstate_to_np(msg)
            )

    res = {}
    for k, v in buf.items():
        if not v:
            continue
        arr = np.stack(v)
        res[k] = _delta_xyz(arr) if arr.ndim == 3 else arr
    return res


def load_all_bags(files: List[str] = BAG_FILES, folder: str = FILE_PATH):
    data = {}
    for fname in files:
        full_path = os.path.join(folder, fname)
        if not os.path.isfile(full_path):
            print(f"[warn] '{full_path}' not found → 건너뜀")
            continue
        key = os.path.splitext(fname)[0]
        data[key] = extract_single_bag(full_path)
    return data



#########################################################################
# 3) 공통 헬퍼 : 3‑D 등축
#########################################################################
def set_axes_equal(ax):
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = np.array([xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]])
    ctrs  = np.array([(xl[1]+xl[0])/2, (yl[1]+yl[0])/2, (zl[1]+zl[0])/2])
    half  = spans.max()/2
    ax.set_xlim3d(ctrs[0]-half, ctrs[0]+half)
    ax.set_ylim3d(ctrs[1]-half, ctrs[1]+half)
    ax.set_zlim3d(ctrs[2]-half, ctrs[2]+half)


#########################################################################
# 4‑1) Trajectory + Error Figure 제작
#########################################################################
def make_traj3d_fig(data: Dict):
    planes, cats = ["xy","xz","yz"], ["robot_poses","desired_robot_poses","tracker_poses"]
    colors = {"robot_poses":"tab:blue","desired_robot_poses":"tab:orange","tracker_poses":"tab:green"}
    fig = Figure(figsize=(9,3.3))
    for i, p in enumerate(planes,1):
        ax = fig.add_subplot(1,3,i,projection="3d")
        for c in cats:
            d = data[p][c][:,2,:3]; d -= d[0]
            ax.plot(d[:,0],d[:,1],d[:,2],color=colors[c],lw=1,label=c.replace("_"," "))
        ax.set_title(p.upper(),fontsize=9); ax.legend(fontsize=6); ax.grid(True)
        set_axes_equal(ax)
    fig.tight_layout(); return fig


def make_traj2d_fig(data: Dict):
    planes, idx = ["xy","xz","yz"], {"xy":(0,1,2),"xz":(0,2,1),"yz":(1,2,0)}
    cats_line = {"desired_robot_poses":"tab:orange","tracker_poses":"tab:green"}
    fig, axs = plt.subplots(1,3,figsize=(9,3.3),squeeze=False); axs=axs[0]
    for ax,p in zip(axs,planes):
        ix,iy,ic = idx[p]; all_x,all_y=[],[]
        for c,col in cats_line.items():
            a=data[p][c][:,2,:3]*100; ax.plot(a[:,ix],a[:,iy],color=col,lw=1.2,label=c.replace("_"," "))
            all_x.append(a[:,ix]); all_y.append(a[:,iy])
        a_r=data[p]["robot_poses"][:,2,:3]*100
        pts=a_r[:,[ix,iy]]; segs=np.stack([pts[:-1],pts[1:]],1); vals=np.abs(a_r[:-1,ic])
        lc=LineCollection(segs,cmap="viridis",lw=2); lc.set_array(vals); ax.add_collection(lc)
        all_x.append(pts[:,0]); all_y.append(pts[:,1])
        all_x,all_y=np.concatenate(all_x),np.concatenate(all_y)
        cx,cy=(all_x.max()+all_x.min())/2,(all_y.max()+all_y.min())/2
        half=1.1*max(all_x.ptp(),all_y.ptp())/2
        ax.set_xlim(cx-half,cx+half); ax.set_ylim(cy-half,cy+half); ax.set_aspect("equal","box")
        ax.set_title(p.upper(),fontsize=9); ax.grid(True); ax.legend(fontsize=6)
    fig.tight_layout(); return fig


def make_error_fig(data: Dict) -> Figure:
    planes = ["xy", "xz", "yz"]                     # subplot 순서
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), squeeze=False)
    axes = axes[:, 0]                               # 1‑D flatten

    for ax, p in zip(axes, planes):
        r = data[p]["robot_poses"][:, 2, :3]        # (N,3)
        d = data[p]["desired_robot_poses"][:, 2, :3]
        N = min(len(r), len(d))
        err = np.linalg.norm(d[:N] - r[:N], axis=1)*100
        ax.plot(err, color="tab:red")
        ax.set_title(f"{p.upper()}  ‖Δp‖", fontsize=10)
        ax.set_ylabel("[cm]", fontsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("time [index]", fontsize=8)
    fig.tight_layout()
    return fig


#########################################################################
# 4‑2) Joint Figure 제작
#########################################################################
RIGHT_JOINTS = [
    "R_Shoulder1_Joint","R_Shoulder2_Joint","R_Shoulder3_Joint",
    "R_Armlink_Joint","R_Elbow_Joint","R_Forearm_Joint",
    "R_Wrist1_Joint","R_Wrist2_Joint"
]
IDX_R = [JOINT_IDX[j] for j in RIGHT_JOINTS]


def make_joint_fig(robot, desired, plane, error=False):
    t = np.arange(robot.shape[0]); data_a=robot[:,IDX_R]; data_b=desired[:,IDX_R]
    if error: data_a=(data_b-data_a)*180/np.pi; data_b=np.zeros_like(data_a)
    fig = Figure(figsize=(8,6)); axs=fig.subplots(4,2,sharex=True).flatten()
    for ax,j,a,b in zip(axs,RIGHT_JOINTS,data_a.T,data_b.T):
        ax.plot(t,a,color="tab:pink" if error else "tab:blue",lw=1)
        if not error: ax.plot(t,b,color="tab:red",ls="--",lw=1)
        ax.set_ylabel(j.replace("_Joint","").replace("_"," "),fontsize=7); ax.grid(True)
    axs[-1].set_xlabel("time [idx]"); title=f"{plane.upper()} - Right-arm "
    title+=("Error (deg)" if error else "Robot vs Desired")
    fig.suptitle(title,fontsize=11); fig.tight_layout(rect=[0,0,1,0.95]); return fig


#########################################################################
# 5) Qt 창 클래스들
#########################################################################
class PoseWindow(QMainWindow):
    def __init__(self, data_dict):
        super().__init__()
        self.setWindowTitle("R‑hand Pose Trajectories & Error")
        tabs=QTabWidget(); self.setCentralWidget(tabs)
        tabs.addTab(self._wrap_canvas(make_traj3d_fig(data_dict)), "3-D Δ")
        tabs.addTab(self._wrap_canvas(make_traj2d_fig(data_dict)), "2-D Δ")
        tabs.addTab(self._wrap_canvas(make_error_fig(data_dict)),  "Position Error")
    def _wrap_canvas(self, fig):
        can=FigureCanvas(fig); w=QWidget(); v=QVBoxLayout(w); v.addWidget(can); return w


class JointWindow(QMainWindow):
    def __init__(self, plane, robot, desired):
        super().__init__()
        self.setWindowTitle(f"{plane.upper()} bag - Right-arm joints")
        tabs=QTabWidget(); self.setCentralWidget(tabs)
        tabs.addTab(self._wrap_canvas(make_joint_fig(robot,desired,plane,error=False)), "Robot vs Desired")
        tabs.addTab(self._wrap_canvas(make_joint_fig(robot,desired,plane,error=True)),  "Error")
    def _wrap_canvas(self, fig):
        can=FigureCanvas(fig); w=QWidget(); v=QVBoxLayout(w); v.addWidget(can); return w


#########################################################################
# 6) main
#########################################################################
def main():
    data = load_all_bags()
    if not data:
        print("No *.bag files found."); return

    app = QApplication.instance() or QApplication(sys.argv)

    # Pose 창
    pose_win = PoseWindow(data); pose_win.resize(1200, 900); pose_win.show()

    # Joint 창들
    joint_wins=[]
    for plane in ["xy","xz","yz"]:
        if {"robot_joints","desired_joints"}-data[plane].keys():
            continue
        r, d = data[plane]["robot_joints"], data[plane]["desired_joints"]
        win = JointWindow(plane, r, d); win.resize(950, 650); win.show()
        joint_wins.append(win)

    if not joint_wins:
        print("joint 데이터가 없습니다.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
