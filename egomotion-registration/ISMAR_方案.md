# 去掉自身运动后再注册三维模型

目标会议：IEEE ISMAR。痛点只有一个：导航用的三维模型注册被观察者的自身运动带着漂。

## 问题

AR/MR 导航要把术前或术中模型画在当前画面上。每帧测到的是模型到相机的位姿

```
T_cm(t) = T_cw(t) @ T_wm
```

`T_cw` 是头显或内窥镜的自身运动。`T_wm` 才是模型相对患者/世界的注册，导航要的是它。

现有 FreeReg 的 SE(3) Kalman（`E:\Free_coordinate_MR_Registration\src\se3_kalman_filter.py`）直接平滑 `T_cm`。相机一加速，滤波器就把转头当成模型在动。出光时刻若再用曝光时刻的位姿，叠影等于角速度乘以延迟。慢速时这项常常小于 PnP 噪声；快速转头时它变成十几毫米的模型错位。

ISMAR 要的证据不是“又一个位姿滤波器”，而是：快速自身运动下，模型在患者坐标系里的误差下降，并且屏幕上的重投影不要变差。

## 方法

只改注册所在的坐标系，不改 FreeReg 的 PnP / ICP。

1. 用静态背景点做 PnP，得到 `T_cw`。模型点另做一次 PnP，得到 `T_cm`。对应关系仍用现有 `src/pose_estimation.py` 的 `pnp_ransac`。
2. 去掉自身运动：`T_wm = T_cw^{-1} @ T_cm`。这一项在模型不动时应当几乎不变。
3. 只在世界系里按小步长更新 `T_wm`。不把相机速度写进模型状态。
4. 出光时刻只外推相机：`T_cw(t+τ) = T_cw(t) @ Exp(ξ_ego · τ / Δt)`。渲染位姿 `T_cm = T_cw(t+τ) @ T_wm`。

实现：`E:\Free_coordinate_MR_Registration\src\egomotion_free_registration.py`。  
对照实验：`E:\Free_coordinate_MR_Registration\scripts\run_egomotion_free_reg.py`。

三种渲染位姿：

| 名称 | 做什么 |
| --- | --- |
| delayed | 用曝光时刻的模型位姿直接画。这是延迟造成的漂。 |
| coupled | 把混合后的 `T_cm` 按匀速外推 τ。这是“整段位姿一起预测”。 |
| egomotion-free | 只外推相机，模型留在世界系。 |

## 受控实验结果

条件写在结果文件里：30 fps，τ = 60 ms，像素噪声 σ = 1.2 px，89 帧，模型固定。数字来自

`outputs/egomotion_free_registration.json`

模型点平均距离（ADD，毫米），越小越好：

| 方法 | 全部帧 | 慢转头 <0.8 rad/s | 快转头 ≥1.6 rad/s（15 帧） |
| --- | ---: | ---: | ---: |
| delayed | 7.36 | 4.86 | 14.94 |
| coupled | 7.38 | 7.76 | 5.46 |
| egomotion-free | 2.06 | 2.03 | 2.19 |

重投影（像素）：delayed 3.63，coupled 1.51，egomotion-free 1.55。

读法：只把混合位姿往前推，屏幕误差会下降，但快转头时模型仍偏 5.5 mm。先去掉自身运动，快转头时模型误差留在 2.2 mm，像素和 coupled 同一档。延迟位姿在快转头时模型误差约 15 mm。

这是合成轨迹加 PnP，不是头显实测。不能写成临床导航已经可用。

## 接到 FreeReg 的方式

某一帧已经有模型位姿和相机位姿时：

```python
from src.egomotion_free_registration import EgoMotionFreeRegistration

reg = EgoMotionFreeRegistration(alpha=0.35)
out = reg.update(T_cm, T_cw, dt=1/30, tau=0.060)
T_render = out["T_cm_render"]   # 送给现有 render_model / composite_ar
T_model = out["T_wm"]           # 患者坐标系里的注册
```

`alpha` 只平滑世界系注册。相机预测不经过这个系数。

## ISMAR 稿只保留这一件事

标题方向：Ego-motion-free model registration for AR navigation under fast viewer motion。

贡献只写三条：

1. 把导航注册从 `T_cm` 改到 `T_wm`，使转头不再进入模型状态。
2. 出光时刻只预测相机扭转，不预测模型。
3. 在受控快转头下，模型 ADD 从约 15 mm 降到约 2 mm；像素误差相对延迟位姿下降，相对“混合位姿外推”持平。

实验下一跳，而不是再加模块：用已有 FreeReg 视频，把背景跟踪当作 `T_cw`，模型 PnP 当作 `T_cm`，按角速度分箱报 ADD 和重投影。没有这段真实序列之前，不要写头显用户实验。

不成立的条件：背景和模型一起动；`T_cw` 本身跟踪失败；τ 未知且偏了一倍以上。这些写进局限性。
