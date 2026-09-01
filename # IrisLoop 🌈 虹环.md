# IrisLoop 🌈 虹环

&gt; **AI 驱动的 MEMS 激光投影自适应闭环系统**
&gt;
&gt; *See the Light, Perfect the Light.*

IrisLoop 是一套将**原生多模态 AI** 与 **MEMS 激光投影**深度融合的交互式内容生成系统。系统通过 USB 摄像头实时采集物理表面的投影效果，利用 AI 视觉分析自主评估差距，并迭代优化投影内容，实现"生成 → 投射 → 观察 → 优化"的全自动闭环。

## ✨ 核心特性

- **视觉自主反馈**：AI 自己"长眼睛"，通过摄像头观察实际投影效果
- **原生多模态理解**：统一理解文本、图像、视频，避免语义漂移
- **MEMS 激光投影适配**：专有位图→MEMS 坐标流转换
- **物理表面自适应**：自动评估不同材质的投影表现并优化
- **数据飞轮**：历史迭代记录持续积累，形成自我进化

## 🚀 快速开始

```bash
pip install -r requirements.txt
export MOONSHOT_API_KEY="sk-your-key"
python -m irisloop.analyzer