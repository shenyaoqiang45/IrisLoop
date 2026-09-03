# IrisLoop 开发者增长计划（AI + 投影）

品味已定：骑行归另一销售团队；IrisLoop 只打开发者可玩 **AI + 投影**。你本人是**品牌体验官**——在 X 定期公开所思所想与实机迭代，GitHub 落代码，Play Kit 收转化。不是营销部矩阵号，是 Build in Public。

| 角色 | 渠道 | 节奏 | 红线 |
| --- | --- | --- | --- |
| 品牌体验官 | X：所思所想 + 实拍 | 持续迭代 | 骑行 ≠ IrisLoop KPI |

**分工红线：** IrisLoop 不背骑行 10 万台。你的工作是让开发者看见一个人在认真玩光与 AI：想、做、晒、改。北星仍是 Play/Dev Kit 订单与复现/投稿；声量来自你的持续在场，而不是投放。

---

## 0. 品牌体验官操作系统

仓库目录：

- `experience/field-notes` — X / 对外「所思所想」草稿
- `experience/laser-log` — 激光实拍素材索引与说明
- `experience/iteration-log` — 迭代日志（`private/` 不入库）
- `hub/submissions` — 投稿规范与待审
- `hub/featured` — 每周精选

| 维度 | 怎么做 |
| --- | --- |
| 你是谁 | 不是广告账号，是第一玩家：亲手跑闭环、踩坑、换模型、改提示词，再把判断说清楚。 |
| X 写什么 | 所思（为什么这样选模型/介质）+ 所见（激光实拍）+ 所改（下一轮迭代）。允许不完美。 |
| 和代码的关系 | 每条有分量的想法，尽量落成 commit / demo / benchmark；X 是前台，IrisLoop 仓库是履约。 |

### 节奏

| 节奏 | 建议 | 目的 |
| --- | --- | --- |
| 每周 3–5 条 X | 2 条所思短帖 + 1–2 条 Laser Shot + 可选 1 条问答/转发投稿 | 可持续，不烧尽 |
| 每两周 1 个可见迭代 | README / 脚本 / 评测有一条可指认的进步 | 证明「不断迭代」不是口号 |
| 每月 1 次长线程 | 阶段复盘：AI 理解、1bpp 规律、Play Kit 反馈 | 沉淀信任，便于新人追更 |
| 固定栏目名 | 如 Field Notes / Laser Log（英文为主） | 粉丝知道来这里看什么 |

**体验官文风：** 少喊口号，多给判断：「这轮 K3 看对了剪影 / 看错了条纹」「生成端过碎所以改 I2V」。读者要的是你的品味与现场，不是品牌稿。卖货自然出现在「想复现就 Buy Play Kit」，不每条硬推。

---

## 1. 两条业务线，两套北极星

| | **IrisLoop（本计划范围）** | **骑行零售（另一团队）** |
| --- | --- | --- |
| 人群 | AI / 视觉 / 嵌入式 / 创意编程开发者 | 骑行用户、渠道、礼品 |
| 产品 | 可黑的激光投影平台 + 开源闭环 | AR HUD 导航与骑行数据 |
| 卖点 | 生成 → 投射 → 观察 → 优化 | 不低头、安全、体验 |
| 渠道 / SKU | Play Kit / Dev Kit / 2-Pack Lab | 独立站骑行页、经销、KOL |
| 北星 | 带 `utm=github\|x` 的开发者支付单 + 月活跃复现/投稿数 | 年销 10 万台等零售指标 — **不写入 IrisLoop OKR** |

---

## 2. 一句话定位（只对开发者）

> IrisLoop: an open playground for AI + MEMS laser projection — generate content, project light, see the result, improve in a loop. Buy IrisGreen Play Kit to hack the beam.

对内中文：**AI 可玩的激光投影闭环。** 不讲码表，不讲卡路里，不讲爬坡。

---

## 3. 对标 Microduck：只抄「开发者爆款」部分

| 可玩 | 可晒 | 可传 | 可买 |
| --- | --- | --- | --- |
| 没买也能跑一部分 | 激光实拍天然完播 | 动画/策略可投稿分享 | Play Kit 冲动价 + 清晰交期 |

| Microduck | IrisLoop 开发者版怎么做 | 明确不做 |
| --- | --- | --- |
| 软件先于高潮交付 | 桌面预览、示例流、`benchmark/groundtruth`、一键推图公开 | 拿骑行 App 当主入口 |
| $399 级自费冲动 | Play/Dev Kit 锚定清晰开发者价与交期 | 用骑行套装折扣话术带货 |
| 技能可分享 | IrisLoop Hub：投影动画、生成策略、VU prompt 可提交 | 封闭应用商店 |
| 短视频完播 | 只发 AI→光 的可见过程：生成、二值化、投射、实拍、再优化 | 夜骑安全广告（交给骑行团队） |
| 社区零广告获客 | GitHub + X 英文；开发者 KOL / HF 系 / 视觉开源圈 | 骑行博主矩阵（非 IrisLoop KPI） |
| 多台更好玩 | 双机对投、Lab 2-Pack、黑客松同场 | 车队团购话术 |

---

## 4. 开发者飞轮

公开可玩资产 → X/GitHub 曝光 → fork 复现 → 买 Play Kit → 产出 AI 投影 demo → Hub 精选再分发。

每一环只服务开发者购买与创作，不为骑行线索优化落地页。

| 环节 | 周稳态目标 | 产出物 |
| --- | --- | --- |
| 造物 | ≥1 个可复现 AI+投影资产 | 脚本、notebook、benchmark 用例 |
| 发声 | ≥5 条英文短内容 | Laser Shot / Build Log |
| 互动 | Issue/PR 24h 响应 | 开发者信任 |
| 转化 | Play Kit UTM 可追踪 | 支付单 |
| 复利 | 每月 ≥4 条精选投稿 | Hub → X |

---

## 5. 90 天（开发者线）

### Day 0–30：买了就能玩 AI+投影

| 事项 | 完成标准 |
| --- | --- |
| README 英文首屏 | 15s GIF：提示词/生成 → 投射 → 摄像头 → 优化；大按钮 Buy Play Kit |
| 没买也能玩 | 本地预览 / 示例 bip 流 / benchmark still 公开；标明「接真机解锁闭环」 |
| Play Kit SKU | 独立站开发者页：主机 + 线材/支架 + 快速入门 + 示例；与骑行零售页分离 |
| 黄金 Demo×3 | ① AI 生成剪影投射 ② K3/多模态看实拍再改 ③ Hello World 推一张图 |
| 归因 | `github` / `x` / `hub` 全链路 UTM；骑行页不混用同一 CTA |

### Day 31–60：Microduck 式打穿开发者注意力

| 战役 | 动作 | 成功信号 |
| --- | --- | --- |
| Launch Week | v0.x Release + X 英文线程 + Show HN / 相关 subreddit | Star 跳升 + Play Kit 订单可见 |
| `#ProjectTheLight` | 主题：AI 生成可投影动画 / 闭环改进前后对比；奖品 = Play Kit | ≥50 份有效开发者投稿 |
| 种子机 | 寄给 AI / 视觉 / 开源硬件创作者（不是骑行博主名单） | ≥10 条公开技术向 demo |
| Hub v0 | 投稿格式 + 每周精选 | 外部 repo 开始依赖 IrisLoop |

### Day 61–90：把「可玩」做成习惯

| 事项 | 动作 |
| --- | --- |
| 教程月 | 周末项目：用多模态改 1bpp 投影；VU benchmark 复现赛 |
| Lab 套装 | 验证 2-Pack 是否提升客单（对投 / 对比实验） |
| 交接资产 | 把可复用实拍包给骑行团队（可选），但不改 IrisLoop 叙事 |
| 复盘 | 只看：开发者 UV、Play Kit 转化、复现率、投稿数 |

---

## 6. GitHub 履约 × X 体验官现场

### 仓库（履约）

- 首屏：AI + laser loop + 实拍 GIF + Buy Play Kit
- Quickstart：5 分钟推图；AI 模式需 API Key
- Benchmark：groundtruth + 你评过的结论可链到帖子
- Changelog：跟你的迭代叙事同步
- **不放骑行卖点**

### X（体验官）

- Field Notes：所思所想、选型、审美判断
- Laser Log：实拍短视频，过程可见
- Iteration：这周改了什么、为何改
- 偶尔 Play Kit CTA，不刷屏
- 第一人称；英文为主，关键判断可中英

### 所思主题库（可轮换）

| 主题 | 示例角度 |
| --- | --- |
| 模型 | 理解优先 vs 生图；为何这轮只用 still_01 |
| 介质 | 1bpp / 绿激光下什么形状能活 |
| 闭环 | 3 轮内是否收敛，还是横跳 |
| 工具链 | BLE 推图、曝光、benchmark 怎么用 |
| 社区 | 回复一个有意思的 Issue / 转发一次真复现 |

---

## 7. 与骑行团队的接口（唯一协作点）

| 协作 | IrisLoop 提供 | 骑行团队负责 |
| --- | --- | --- |
| 品牌可信 | 开源与硬核 demo 背书 | 零售转化与渠道 |
| 素材复用 | 可选：授权技术实拍 | 改写成骑行广告（自行剪辑） |
| 线索 | 不收集骑行线索 | 自有投放与销售漏斗 |
| SKU | Play/Dev Kit 定价与文档 | Retail SKU 与促销 |

---

## 8. KPI（IrisLoop 专用）

| 注意力 | 参与 | 收入（开发者） |
| --- | --- | --- |
| Stars / Forks / 克隆 | Issue / PR、Hub 投稿 | Play/Dev Kit 支付单 |
| X 完播与开发者互动 | benchmark 复现报告 | `utm=github\|x\|hub` 占比 |

**北星：开发者支付订单数。** 不要用骑行整机销量考核 IrisLoop；也不要用纯 Star 替代订单。

---

## 9. Microduck 式发布周（开发者）

| 时间 | 动作 |
| --- | --- |
| T-14 | 公开预览 + benchmark + 英文 README；Play Kit waitlist |
| T-7 | AI / 开源种子机到手；5 条 15s AI→光 视频就绪 |
| T-0 | Release + 开发者购买页 + 官方 X 英文长线程 |
| T+1～3 | 只转二创与复现；每天 Laser Shot |
| T+7 | Hub 投稿规则与第一期精选 |
| T+30 | 复盘 Play Kit 转化；决定是否上 2-Pack Lab |

---

## 10. 本周启动（体验官版）

1. X 简介改成：IrisLoop brand experience — AI + MEMS laser projection
2. 置顶一条：你是谁、在玩什么闭环、如何 Follow / Buy Play Kit
3. 定栏目名 Field Notes / Laser Log，本周发出第 1 条所思 + 1 条实拍
4. README 英文首屏与你的叙事对齐（人在迭代，不是冷冰冰 SDK）
5. 购买链指开发者 Play Kit 页
6. 建私人迭代日志（可私密）：日期 / 假设 / 实拍 / 结论 → 供发帖取材
7. 周复盘只看三件事：发帖是否诚实、仓库是否跟上、有无开发者复现

---

## 品味落成一句话

你是 IrisLoop 的品牌体验官：在 X 上持续公开所思所想与激光现场，在仓库里不断迭代履约。开发者因信任一个真玩家而来；骑行销量交给另一支队伍。
