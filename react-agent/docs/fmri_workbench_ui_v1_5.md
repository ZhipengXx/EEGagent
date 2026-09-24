# fMRI 检查工作台 UI V1.5

这一版只改展示和只读适配，以及按需导出单帧脑图。检查判决、阈值、TRIBE 预测和 DeepSeek 调用都没有改。

## 启动

在 `react-agent` 目录：

```bash
.venv/bin/python -m react_agent.fmri.workbench
```

浏览器打开 http://127.0.0.1:8765 。

## 文件

- `src/react_agent/fmri/workbench.py`：本机页面和只读接口。
- `src/react_agent/fmri/workbench_view.py`：把 report、事件、用量和记忆收成一份 view model。
- `src/react_agent/fmri/metric_defs.py`：指标释义。公式按现有工具代码写。
- `src/react_agent/fmri/workbench_static/`：页面、样式和脚本。没有外网 CDN。
- `src/react_agent/fmri/_brain_strip.py`：单帧按 left / right / posterior 分文件导出，图内不再写英文 signed 标题。缓存键是 `frames/frame_v2/{mode}/{index}/{view}.png`。旧的 `frame_v1` 不会再被页面引用。

## 和上一版页面的差别

- 新建检查收进右侧抽屉。默认仍是 image16 的 rule 配置。
- 左栏显示样本名、文件时间和判决标签，不再把长路径当标题。
- 主区是概览、脑图与时序、指标解释、Agent 轨迹、产物与记忆。不再把整份 `workflow.html` 嵌进 iframe。
- 布尔、类别和覆盖状态用文字标签。`described` 是信息色，`unassessed` 是中性灰。
- `evidence_confidence`、训练权重和费用为空时显示未提供。`0` 和 `否` 保持原值。
- 脑图默认单帧。滑块请求真实的那一帧。拼图仍可切换。没有数组时只显示拼图，并写明只有拼图。

## 公式来源

- `overall_delta_rms`：`sqrt(mean((pred-gray)^2))`，等顶点。见 `tools/control_contrast.py`。
- `normalized_delta_over_control_rms`：上式除以 `sqrt(mean(gray^2))`，分母大于 `1e-12`。
- `temporal_change_ratio`：`mean(|diff(spatial_mean)|) / (std(spatial_mean)+1e-8)`。见 `tools/numeric.py`。
- `max_frame_diff_ratio`：空间均值相邻变化的最大/中位，分母加 `1e-8`。
- `max_step_over_median`：contrast RMS 相邻步长 A 的最大/中位，不是顶点模式变化 S。
- 这些字段都没有在界面里补正常范围。

## 验收

用 accordion 的 `accordion_01b_f1f17445` 读出 view model：`finite_ratio=1`，contrast 时序合同为否，置信度未提供，接受计划数为 1，费用未提供。单帧接口会换成不同帧的图。

1440 和 1280 宽度在 `app.css` 里收成两列卡片，1280 以下三个脑图改为上下排列。768 及更窄（含 390）侧栏改为抽屉，卡片改成一列。这次没有可用的浏览器自动化，也没有本机 Chrome，所以没有保存这四档截图。页面数据用 accordion 与 antenna 的只读接口核对过。

## V1.5.1 对齐

这一轮只收紧现有页面的对齐和信息层级，不改判决，也不重跑 TRIBE 或 DeepSeek。

- 顶栏只留产品名、短状态和「新建检查」。样本名只在内容区出现一次。
- 侧栏按 `sample_id` 分组，组内按 `report.json` 修改时间从新到旧。点哪一条就打开那条 `run_id`。次行写「文件修改时间」。列表判决用短句「配置内通过」，完整句子留在概览卡，并放在 title 里。
- 概览首行是样本名、形状和一句协议摘要。只有 metadata 确认是 image16 时才写「16s · 4s 灰 / 1s 图 / 后续灰 · fsaverage5」。fps、t_stim、analysis_goal 放在「实验与生成信息」里。四张卡是数值检查、问题覆盖、检查深度、调用成本。卡片下面不再复述同一句结论。
- 问题行用中文名。内部 id 在展开处。后续检查看报告里的 followups；列表为空时写「无待跟进问题」，不从 answered 推断。刺激图缩略图只在图片位于该 run 目录、能被受控路径读到时出现。
- 脑图工具分成信号、播放、适应/全屏/单帧三组。三个视图等宽，`object-fit: contain`。色阶在图旁，写明模型输出单位，以及红/蓝只表示正负。时间轴只给 image16 标出灰屏、图像、灰屏，并标 t=4s 和 t=5s。
- 指标按输入、分布、灰屏对照、时序、空间、模型参考、运行成本分组。比例 1 显示为 100%，计数加千分位，shape 显示成「16 个时间点 × 20,484 个顶点」，分位数显示 P5 / 中位数 / P95。0、否、未提供保持原样。正负不涂红绿。行点击或 info 图标打开原来的解释抽屉。
- Agent 轨迹和产物页沿用同一套标签、容器和按钮高度，事件生成没有改。

## 帧和视图一起换

滑块、上一帧和播放不再逐张换图。left、right、posterior 都载入后，说明文字、三张图和曲线上的当前帧一起更新。加载时保留上一组已经齐的图，并写「正在载入第 n 帧」。过期的请求不会再画上去。播放会等当前帧到齐再进下一帧，并预取相邻一帧。

已有 `frame_v2` 图片时，单帧接口直接返回文件，不再重建报告或重算色阶。同一帧的三个视图只渲染一次。色阶在三视图右侧，窄屏改到图下方。

## 限制

- 时间轴只有 image16 且 T=16 时写成绝对秒和 `t_stim = t-4`。其他协议不编造 16 秒。
- 视角名称是渲染器的 left / right / posterior。
- raw 和 contrast 色阶分开，页面写明不能直接比颜色强弱。
- 事件没有 `parent_call_id` 时轨迹是按时间排的平面列表。
- 没有取消任务的接口。
