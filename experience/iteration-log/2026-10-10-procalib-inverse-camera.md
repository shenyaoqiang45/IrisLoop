# 2026-10-10 相机-投影仪逆标定：排障与方案定型

海康 MV-CA013-A0UM（USB3，Mono8 1280×1024）× IrisGreen 激光投影（第二台个体
`20:6E:F1:81:71:62`，原名 Iris-G-817162）联合标定。目标产物：相机内参 +
投影仪逐像素光线场。代码：`irisloop/procalib.py`（camint/collect/solve/verify）。

## 硬件接入（MVS SDK 5.1.0）

- 静默安装：`MVS_STD_5.1.0_260909.exe /S`，须经 `cmd //c` 调用（zip 解压出的 exe 缺执行位）。
- 运行时 DLL 在 `C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64`
  （**不在** MVS 安装目录），导入 Python 绑定前要把该目录加进 PATH。
- 绑定层结构变了：无 `MvCameraControl_header.py`，改为 `CameraParams_const/header.py`；
  设备结构体字段是 `SpecialInfo`（大写 S）；无 `MV_TRIGGER_MODE_OFF` 常量（传 0）。
- 枚举混排 GigE 与 USB3 设备，且**序号跨次枚举不稳定**——必须按序列号打开。
  对 GigE 设备 OpenDevice 报 `0x80000201`（地址不可达），相机被 MVS 客户端占用报
  `0x80000203`（先关 MVS.exe）。
- 封装：`irisloop/hik_camera.py`（接口对齐 UsbCamera）；冒烟 `tools/hik_smoke.py`。

## 标定板 CB-0300-W-001（圆点阵，非棋盘格）

- 11×9 圆点、间距 25mm、5 个大点定向基准；布局解析产物 `cbinfo/CB-0300-W-001.layout.json`。
- 检测：自适应阈值（黑点比局部背景暗 C=8 灰度级）——固定阈值在全局变暗时直接归零，别用。
- 点阵匹配：BFS 图游走 + 单应重分配 + 基准点 4 旋转锚定（`match_lattice`）。
  教训两条：
  - 全局仿射拟合在透视下边缘超差（最外两列会被残差阈值吃掉）——透视必须用单应。
  - 11×9 非方网格的相对帧可能落在任一物理轴上，**90° 旋转也要搜**（不是只有 180°）。
- 产品照网格方向与物理方向不一致：解析布局时按"列=11 点长方向"规范化，否则基准点帧维度不自洽。

## IrisGreen 投影的设备特性（本次最贵重的经验）

1. **UI 状态机会吞内容**：BLE 连接后数秒内显示「连接中/连接成功」状态屏，
  play 命令会被覆盖甚至吞掉。**一切投影内容都要用相机闭环验证**（推图后拍到预期内容才算数，
  见 `_project_verified`）。靠固定 sleep 等时序是靠不住的。
2. **扫描占空比**：曝光 < 激光扫描整帧周期（~16-20ms）会漏检点阵；曝光 ≥ 40ms 整帧积分，
  条纹自然消隐，**不需要短曝光压条纹**——全程 40ms + 增益 8dB 一档曝光搞定。
3. **同次播放内投影位置极稳**（帧间 0.2-1px）；但状态屏/内容切换帧混入多帧平均会
  "重影"（两套点阵叠在一张图里）——**激光帧用单帧，不要多帧平均**。
4. **双振镜 MEMS 不是针孔器件**：水平/垂直各有独立扫描中心。针孔模型闭环失败
  （Zhang 闭式解 fy/fx=1.48、calibrateCamera 陷入坏盆地 cx=1661）。
  **正解：逐像素光线场**——同一投像素点跨姿态的 3D 点天然共线，PCA 稳健拟合出
  每条光线（origin+dir），三角化时对光线场双线性插值（`fit_ray_field` /
  `ray_field_lookup` / `triangulate_rayfield`）。

## 数据分析排障记录（run1 数据集）

- 相机内参：15 视角 × 99 点，RMS 0.203px，jackknife 焦距 std 1.5px —— 可信。
- "同一投像素点 ⇒ 同一相机像素"的刚性不变量**是错的**：板深度变化时，点在相机画面里
  沿**极线**滑动（本装置基线近似水平 → 几乎纯水平移动，±20px 完全正常）。
  正确的刚性监控是 3D 线一致性：新姿态的 3D 点到既有像素光线的距离中位数（>3mm 作废）。
- 姿态内板移动（板帧与激光帧之间板子动了）对"逐姿态单应"完全不可见，但会污染 3D
  点 —— 板在姿态内必须绝对静止（台钳夹紧，人等 1 秒再采）。
- 板侧链路（对应关系/板位姿/3D 点）交叉验证达 0.1mm 级；失真大头在投影侧散斑质心
  （跨姿态 ~2mm 级），靠多姿态稳健拟合吸收。

## 采集工艺（collect 最终形态）

1. 连接后推全黑并相机验证灭点（避开状态屏）。
2. 5 个锚点逐个投影，相机轮询确认恰好 1 点 → 粗单应 H_boot。
3. 预览循环（板点检测叠加）；空格采集一个姿态：板帧（当前预览帧）→ 推点阵
  （闭环验证稳定）→ 单帧激光帧 → 恢复全黑（闭环验证）→ 匹配 + 3D 线一致性检查。
4. 12–16 个姿态；r 键清空重采。

## 当前状态（2026-10-10 下班前快照）

### 已完成
- MVS SDK 5.1.0 静默安装 + 海康相机 Python 链路（`irisloop/hik_camera.py`、`tools/hik_smoke.py`，文档 `docs/hik-camera.md`）。
- 标定板 CB-0300-W-001 解析（`cbinfo/CB-0300-W-001.layout.json`），实拍 99/99 点全匹配。
- 相机内参：`captures/procalib/run1/camera_intrinsics.npz`（15 视角，RMS 0.203px，jackknife 焦距 std 1.5px）——**长期有效**，重采不用重跑 camint。
- 完整流水线 `irisloop/procalib.py`（camint/collect/solve/verify），加固版采集：
  闭环投影验证、单帧激光帧、单档曝光 40ms+8dB、3D 线一致性刚性监控、姿态剔除。
- 方案定型：**逐像素光线场**（投影仪非针孔）；文档 `docs/procalib.md`。
- 测试 23 项全绿（含合成装备反演精度、光线场恢复）。

### 未决问题
- **两台 IrisGreen 都没电/休眠**：新机（817162）不广播（休眠或没电），旧机（B6B7CA）
  广播但不出光；且旧机 `get_brightness`=100、`battery_raw`=100% **与实际电量不符**
  （设备电量/亮度读数不可信，也是设备 quirks 之一）。
- run1 点阵数据：部分姿态被状态屏过渡帧/多帧平均涂抹污染，仅作算法验证用。

### 下一步（明天从这里继续）
1. 充电/唤醒投影仪（任意一台；旧机地址是默认地址 `F4:12:FA:B6:B7:CA`，
   新机用 `--address 20:6E:F1:81:71:62`）。先确认镜头有绿光出射。
2. 重采（装置已刚性夹紧，勿动相机和投影仪）：
   ```bash
   python -m irisloop.procalib collect --dir captures/procalib/run2 --poses 12
   ```
   预览窗口里：台钳固定板 → 空格采一个姿态 → 换 12 个姿态。每个姿态内板子绝对静止。
3. 解算 + 验证：
   ```bash
   python -m irisloop.procalib solve --dir captures/procalib/run2
   python -m irisloop.procalib verify --dir captures/procalib/run2
   ```
   合格线：solve 的光线场线拟合 RMS 中位数 <0.5mm；verify 平面 RMSE ~1mm 级。
4. 达标后：三维重建演示（点云），更新 `docs/procalib.md` 实测指标。

### 一键自检（明早开工先跑）
```bash
python -m pytest                    # 软件回归（应 23 passed）
python tools/hik_smoke.py --fps     # 相机出图
python tools/board_check.py --dark  # 板检测（应 99/99）
```
