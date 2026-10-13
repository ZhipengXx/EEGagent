# EEGagent 真实研究修复与停止快照

## 当前停止快照（2026-10-13）

用户已要求停止研究。`eeg_context_real_20261013_r1` 已通过原生stop进入`cancelled/user_cancelled`，control为stop；持续监控和自动修复目标已暂停，无运行中的训练或未结算API。最终累计55次API、6项训练、10222.762955665588 GPU秒（约2.84 GPU小时）；历史费用和实验产物保留。

本代码版本还包括后续完成的原生授权入口：通过追加、可核验的receipt支持候选范围20，以及Planner与闭组memory各一次最多2次API的恢复，原总预算300API/80训练/80GPU小时保持。用户曾明确批准这项续研范围，随后在实际发行授权前要求停止。因此新授权代码及离线回归已完成，但未部署到当前研究runtime、未发行授权receipt、未执行这项续研；真实CAM原goal和candidate上限仍为2。代码提交不会恢复实验。

本地证据为31批XML，处理批次重叠和既有用例改名后577个唯一节点通过；并非当前整仓一次完整测试，也不证明后续长循环稳定或方法性能提升。授权接入的最后批次分别为core25、scope consumer2、Planner39及memory兼容101（包含新增7，不能再相加）。测试、XML、运行目录及机器证据按仓库现有忽略策略留在服务器，本次快照发布代码、prompt和这份报告。

下文是分阶段执行记录；其中运行、预算、部署和授权状态均以该阶段观测为准，当前状态以以上停止快照为准。

目标为既有 `eeg_context_real_20261013_r1`，没有新建 campaign。基线代码为 `4d9d5138e118a32585f84d37031b14169a8b9496`。修复前状态 `blocked:api_operation_attempt_limit`；已使用 25 API、4 项训练、3472.091239929199 GPU 秒，剩余 275 API、76 项训练、79.0355 GPU 小时。

最初执行依据为 `react-agent/docs/EEGagent-run-repair-codex-20261013.txt`。修复阶段集中处理执行缺陷，不扩大总资源预算或科学评价准入；原goal、已训练实现和评价协议保持。用户随后明确要求保存Git版本，才执行代码提交和推送。

## 根因及改动

| 问题 | 修复 |
| --- | --- |
| 纠正反馈在 adapter 丢失，同一坏 proposal 被复用 | canonical parent/child request 带 current errors、坏回复 digest、父 operation/task 和有界草稿/ref；语义字段参与 input/cache identity |
| Phase A/B conditions 示例为 list，validator 要求 object | 示例使用可验证条件对象，程序继承实际 scope/seed/method/selection；旧 list 只读兼容 |
| Phase A/B 发布失败不能有效纠正 | 有限一次纠正；已付费 child 可恢复，失败保留草稿并退避，不提前推进 baseline/coverage |
| 慢生成/纠正超过 lease | owner/source/base 匹配的有限在途续租，默认调用保护600秒；不增加调用额度 |
| 普通角色付费回复后崩溃会重付 | 原冻结请求、response hash、配置、来源和 domain validator 恢复；写入/注册/completion/state 边界均测试 |
| prompt/config 升级混入旧 pending task | 优先恢复合法 paid parent；无回复时显式 child，新输入继承原费用/额度 |
| 两折证据没有单 job_dir，诊断不可用 | 新 suite-aware resolver 使用已验证 score/freeze 的真实 job、现有导出和曲线，写独立派生 artifact |
| 观察过滤后证据数量与执行不一致 | action input epoch 绑定原生科学来源与实际诊断输入文件 hash，原生规划/执行使用同一版本 |
| 诊断丢失、损坏尾行终止读取 | method ancillary 由来源验证产生；有效行保留、损坏行/截断/缺失明确标注，原 history 只读 |
| 筛查已完成但 seed coverage 声称未测 | measured screening seed0 与未测十折 primary 分开；fold数量与独立seed数量分开 |
| 执行/拒绝/detach 重写旧 decision | 追加 decision receipts，兼容旧内嵌状态，原 raw/execution/decision 保留原字节 |
| 已死亡且结算完整 invalid 分支不能解除 | 在 lifecycle 控制层验证死亡与实际结算后 detach，不改 invalid 状态、不退还费用或候选名额 |
| 大 episode/batch needs_sections 无出路 | Phase A 真实字段/字符范围分段，父全部完成后推进 processed；Phase B 连续 prefix/section，未完成不跳 baseline |
| UI 在 receipt 写入、state 更新之间仍显示 pending | view/timeline 只读应用同一 receipt 投影，不写回 state 或原 decision |
| canonical fold 的普通 protocol wrapper 绕过 identity cache | 对小型 wrapper 作有界结构 fingerprint，大 Frozen 叶复用；保留原 JSON hash 算法与结果 |
| 已确认输出截断在首attempt直接退出 | 在原operation allowance内做一次同cap/model/config的精简JSON纠正，带parent call和坏回复digest；二次仍失败即停止 |
| _lazy Designer 有task但Compiler只使用空bound | 将已持久化task身份传入Compiler，normal/repair均禁止该冻结task内的同步/语义维护，共享bound不变 |
| Compiler已经完整读到未知字段仍反复要求读取 | 验证registry中同source/hash/field的完整chars覆盖与原内容相等，literal交付该字段；partial/stale/其他字段不解除义务 |
| 原生失败丢失Compiler具体原因 | 在失败event/state保留有界reason/context_result/budget_report，不改变错误状态或gate |

诊断保持既有 `campaign-history annotation` 成本语义，candidate target 为标注，不偷偷改成只诊断该 candidate。现有导出只代表其实际抽样与 source development 范围，不能当作额外 held-out 实验。新信息可触发既有 Analyst/Curator，后续模型调用另进入原生 API 账本。

## 本地验证

使用 `/tmp/eegagent-minilm-validation-venv/bin/python`、本机 offline hooks、临时 campaign/SQLite/registry 和真实 SDK MockTransport。离线环境禁用真实网络/GPU，不使用当前 CAM 作为可写 fixture。

- 完整 context 回放：260 passed，含100轮、旧164617字符 fixture、维护 native adapter 故障恢复。
- 通用角色恢复及缓存/配置：131 passed，含31项新增实际SDK故障场景。
- 生命周期、阶段覆盖原生 consumer：29 passed，含阴性两折和有条件2+8确认。
- suite诊断闭环：7个独立用例通过，使用真实CPU scorer/verify_score；训练完成和模型回复是明确标注 fixture。
- 阴性两折闭环另复测同一用例：真实 lesson publication validator 接受条件化经验，Planner 收到完整条件，restart 不重复分析或发布。
- identity cache：22 passed；真实 canonical_fold_identity 处理22.79MB临时协议，同scope第二次哈希中位耗时150.9ms降到37µs，SHA一致。这只代表该重复哈希热区，不代表完整 worker 加速倍数。
- UI receipt crash window：3 passed，包含执行/拒绝/detach，页面读取不修改原 state/decision。
- 截断/native广回归：151 passed，包含10项新增actual-SDK同cap纠正/失败restart/额度不扩/未知回复测试。原允许输出调整策略仍按既有goal执行；当前研究未提高cap。
- 最后编译身份补丁：14 passed（4项身份用例+10项截断用例）；涵盖未绑定Designer实际SDK的normal/repair任务ID、manifest与维护flags。
- 后续真实Compiler堵塞：48 passed，含完整字段读取、partial/stale/伪scope和恢复consumer；原输入隔离回放由NeedsContext变Ready，code literal不变。
- 原生失败诊断保留：38 passed，含4项新增原生tick，用于保存具体原因而不改费用、状态或准入。
- 根模块兼容批次初次118 passed/2 failed；可选 job 记录缺失改为 explicit gap，另一个为 mock finish proof/callcount断言。三个对应回归已复测通过。

批次相互重叠，不能直接相加。旧 `test_crash_recovery_followups.py` 前四个 `experiment_not_approved` 失败在原 HEAD 代码独立复现，不为通过它们放宽审批。旧 provider schema title/description 的相等断言更新为 schema-aware 注释压缩，runtime语义保留。

测试报告及命令记录在 `react-agent/version/run_repair_20261013/test_evidence.json`，XML归档于同目录 `test_xml/`。原root批次完整argv未在上下文中保留，索引明确区分实际命令与从XML模块重建的未执行重跑命令。测试证明控制与来源机制，不证明真实模型质量或方法收益。

## 产物保护与部署

CAM 外已记录修复前 diff、状态/账本、关键源 hash、673个受保护文件、22个追加日志前缀及一致性 SQLite backup。checkpoint只记录原路径/hash；未复制 EEG 全数据。

先部署21个未冻结控制/提示模块，安全步骤边界收尾后为29个文件；都沿用该研究原 runtime 绝对路径。另有29个冻结训练/evaluator/协议依赖全部保持原hash。旧决策、请求/回复、spec、checkpoint、history、metrics、score、freeze、manifest和验证receipt在部署前审计均原字节一致。未刷新旧科学manifest hash。

原生预检已确认锁可取得、无未知 API 费用、授权仍属本 CAM；修复后合法动作重新包含 collect/diagnose 和未测试方法的设计。已有四项50epoch训练保持复用，不为验证修复重训。

## 已完成的真实短窗口

原生 `resume --max-ticks 1` 已正常完成，worker PID3531903宿主确认退出，原生记录 `worker_tick_limit`。状态从旧blocked恢复为planning，旧failure已原生清除。新d8为design_experiment，追加执行receipt `decision_receipt_46e70719affa89c0634a1b59`；原decision保留原字节。

真实memory generation `task_6ba42a965709 / b97bc2bafcce` 的数值fact不属于已交付partial section，因此publication rejected。纠正 `task_86c280e27d82 / 8e45bfb25ad0` 确实交付错误、parent/draft引用并使用新身份，新proposal `98ff9208591e…` 验证通过；实际compiler preview没有有效压缩收益，原生no_op而未发布。Snapshot仍revision1、12processed/4abstracted。这次是纠正成功但无收益，并非token溢出或虚假memory发布。

Planner call `2182ecb980ca` 使用135734 application chars（system23811+user111923），生成合法新d8：在c1阴性两折反馈后设计容量匹配的机制对照。Designer `ecf8e1235a11 / task_cf9357ef3236` 到16384输出token上限而截断；新spec `art_7394835dc74a` 保持blocked，没有批准或训练。旧c1仍为真实阴性，不被改成成功。该截断作为本次真实运行新问题记录，持续恢复前补齐既有有限attempt内的纠正路径，不提高额度。

短窗口累计31API/4训练/3472.091239929199GPU秒，无未结算费用；比修复前新增6API、0训练、0GPU秒。剩余269API、76训练、79.0355GPU小时。max_candidates仍2、c1已消费1，工程repair上限仍2；不退还已消费名额。

短窗口后审计673原文件、22日志前缀、29冻结科学文件全部一致。新增102条文件路径及追加日志、合法控制投影变化已分类写入 `artifact_audit_after_resume.json`，其中原API和GPU费用全保留。路径数量为该审计时点，后续持续原生运行将增加。

恢复首次CPU校验约12分钟才发出首API。冻结method manifest约273.5MB，两份suite manifest各553.5MB，协议内重复嵌套query/gallery数据，首次数据身份校验含26.245GB读取。没有将这些巨型manifest原样发给模型；未重写科学manifest。已修正未冻结identity cache对小型普通protocol wrapper失效的问题，冷读/首次校验仍保留。

输出截断与编译身份收尾已通过回归并部署。中间worker3863345在初始恢复阶段通过原生pause安全退出，未发送新API或训练；部署均在锁可取且无在途费用边界完成。随后worker3909294实际触发context_blocked，无新增API、原生退出。具体原因在隔离一致副本重放确认：`/control_source/encoder_reference/excerpt` 为1621字符完整源码，bounded chars read确已覆盖0:1621，但旧consumer只处理declared_functions/lines，不能消费该完整字段读取，130548字符且provider/app预算通过时仍报unclassified。

完整字段读取修复后，隔离同输入Ready、prepared.verify=True、132532chars、hard160000不变、provider_budget_ok=True、字段逐字一致。只接受已注册的原source/hash/pointer/start/end/body相等证明，不为任意excerpt放宽分类；context read仍为delivery-only，不当作原生执行许可。已部署并再次原生 `resume --max-ticks 200 --poll-seconds 30` 继续原研究及原授权。最新worker与后续动作将在恢复完成后追加；不能把短窗口中的blocked设计说成新有效实验。

## 最终持续续跑的真实进度

持续worker PID4063456由宿主确认运行。真实Compiler已经通过，Planner call `90e542a84341` confirmed/stop，输入38221tokens、输出21590tokens（含reasoning12497）。实际交付system23811+user95705=119516字符，wire仍保留完整1621字符baseline编码器字段，没有扩大160000字符上限。

新合法决策d9选`propose_experiment`，以容量匹配检验c1失败的竞争解释，parent仍为c1。决策与原始回复已原生落盘；执行、设计批准、代码检查、训练由持续worker继续处理。不能据此声称候选已批准或性能提升。

截至这次确认，累计32API/4训练/3472.091239929199GPU秒，API无未结算intent；比修复前新增7API，未新增训练或GPU费用。剩余268API、76训练、79.0355GPU小时。实际预算以继续变化的原cost/state账本为准。原生API美元价格仍unpriced，不能把未定价当作免费。

回归证据20份XML按最新node结果去重497通过，不是当前整仓一次批次；最终字段读取补丁之后有48项精准回归，具体时间与覆盖在索引中区分。保护审计仍为673旧文件、22日志前缀及29冻结科学依赖一致，新增/追加/控制投影见机器清单。最终29个未冻结控制文件全部与部署runtime一致。当前HEAD保持原提交，本任务未commit/push。

主要可审查产物：`continuous_resume_evidence.json`、`artifact_audit_after_resume.json`、`test_evidence.json`、`final_control_sources.json`、`repair_code_diff.patch`，均在CAM外的 `react-agent/version/run_repair_20261013/`。只读原输入重放脚本为 `replay_context_block_readonly.py`。

最后宿主及原生投影确认：d9已执行（ok=true/executed=true），state为planning、detail=null，worker4063456持续运行。累计35API，全部已结算，4训练、3472.091239929199GPU秒；剩余265API、76训练、79.0355GPU小时。本次新增10API、0训练、0GPU秒。后续内存维护、设计、实现与训练继续由原生worker处理，费用随账本更新。

## 持续监控：c2筛查与重复摘要计费

EEGagent后续自主批准c2、完成实现/检查/双卡工程probe，执行d11 run_screening。第一折fold07原生完成50epoch和评估；baseline07/10均复用。第二折fold10已由恢复后的worker290629启动，训练PID307539在GPU7/2运行。两折尚未全部完成，未给出c2收益结论。

监控中确证旧summary重试身份的性能bug：native_closed:8/9/10科学batch及previous_snapshot完全相同，base revision=1/source epoch=4，仅控制authority变动，却三次创建新job共花6API。实际一致副本还识别出同batch的9个历史epoch。maintenance/persistence现按同source/base派生family累计尝试/退避，跨turn/authority/prompt/model保留消耗；原request/CAS身份仍完整验证，旧原始proposal/费用/coverage不改。

74项隔离回归通过，含actual SDK旧6付费调用继承与summary defer后PhaseA/B正常发布。实际CAM一致副本的新控制轮返回attempts_exhausted/callback0/calls0，保留9次旧尝试及rev1/sourceEpoch4/12processed/4abstracted。补丁仅两个未冻结模块，在原生pause且第一折训练+评估自然结束、无在途API的安全边界部署，随后原生resume复用第一折推进第二折。详细证据为source_family_repair.json及test_xml/eegagent-source-family.xml。

检查时累计43API，实际启动6项训练、cost投影已settled5项；GPU已结算6856.235030174255秒，第二折仍在进行不能把pending GPU当0费用。剩余API257、未启动训练slot74、GPU剩余额需再扣当前在途耗用。未定价API仍unpriced。

phase_reservations.json最初被保护清单误归为整文件不可变；实际它是native预算投影。仅去掉新增c2 reservation可逐字重构修复前5443bytes及原SHA84094ea7...0726，旧baseline/c1两条完全未改。审计脚本据此验证合法新增投影，不修改原保护基线或真实CAM。

## 本轮修改完成与真实结果（2026-10-13 状态答复）

本轮控制代码修复、针对性回归及安全部署已经完成；最近一次补丁的真实续跑验证仍在进行。原生resume返回worker PID834371，宿主确认进程正在运行；截至核对时原生状态仍保留上次context_blocked诊断，尚未出现新动作，不能仅凭worker启动宣称最后一个堵塞已在真实CAM解除。

最后一个已定位的堵塞并非provider token溢出：Planner三项option的外层/内层evidence_refs不一致，原生需要机械纠正，却将外部option_repair_request误判为normal。147046字符超过normal的144000 fallback，而provider预算通过；同输入repair回放147099字符可合法Ready并verify通过。llm.py现将有真实rejected option/error/draft的原生纠正及有效legacy schema_error归入已有repair模式。normal144000、repair严格小于160000以及provider额度均未提高。新增6项实际SDK mock回归、常规65项回归分别通过；这是离线证据，最后补丁的真实CAM请求尚待观察。

真实c2两折已全部完成50epoch及评估。当前三组两折Top1均值：baseline9.50%，c1为0.25%，c2为1.00%；相对baseline分别为−9.25和−8.50个百分点。c2为容量匹配对照，两折仍明显落后，原生not_promoted、无确认准入。全部为seed0、fold07/10的开发筛查，不是十折或最终测试结果；baseline训练复用，合计实际6项训练。

真实working memory已从revision1推进至2，processed从12至22、abstracted从4至13；新科学batch完成有效摘要发布。相同source/base跨控制轮重复付费问题已有source-family修复及74项隔离回归，PhaseA/B在当前真实CAM中的后续推进仍待验证，不能把fixture publication当真实结果。旧请求/回复、训练源、checkpoint、history和score未由操作者改写。

截至本次核对累计53API、6项训练、10222.762955665588GPU秒（2.839656小时），无未结算API；总授权剩余247API、74训练、77.160344GPU小时。当前冻结goal另有max_candidates=2，c1/c2已用完2个名额，因此剩余总预算不自动授权新的候选。后续原生仍可诊断、总结及完成审计，操作者未提高该限制。

## 后续真实堵塞：Planner轮额度被维护占用

worker834371完成校验后新增5fe06edaea57和dd98c7b34ecc两次summary生成/纠正，实际API到55，均已结算。它们属于同source/base的第三epoch，正确达到family上限3，未归零；publication失败，snapshot2/22processed/13abstracted保持。随后原生planner_round_attempt_limit退出，宿主确认进程不存在。

round_10453cb040b248e460f676c9eb676c4b的6条实际预约全部保留：bdb61acdccfb、3d6da10940e0、a702ce7a0527、656d3872f68d、5fe06edaea57、dd98c7b34ecc；其中5次维护、1次Planner primary。prepared trace9e326aa7daf6为新的primary重入，尚无reservation/intent/收费；真实仍未进入Planner repair。同prompt/model并不足以证明同冻结输入，artifact index等投影有变化，未放宽付费回复采用证明。

两项最小调度修复已经测试并部署：ContextService inline upkeep从共享round真实余量中为Planner保留原有两次操作额度；worker在原startup来源/身份校验后，仅对精确recoverable Planner round-limit阻塞执行已有闭组memory后处理，再保留原block返回。暂停/停止、委托、未知费用、活动任务及全局额度继续生效，不构造Planner、不执行tick/训练，不更改已耗round/candidate/goal。原native_closed turn及两次maintenance上限保留，不靠新turn制造额度。

新增upkeep7项、兼容58项（含100轮回放）、worker27项（21新增+6既有）分别通过；SDK mock的PhaseA/B发布只证明fixture机制，不能冒充当前CAM实际发布。29个冻结科学源仍全部原hash。实际部署仅service.py与worker.py，before/after SHA及一致SQLite备份记录在planner_round_starvation_fix.json、runtime_control_before_planner_quota_fix。

原生run --max-ticks1已启动记忆缺失阶段恢复（exec session85487、worker PID1089774，宿主确认运行）。当前原Planner状态仍blocked、6/6耗尽，观察到API55/6训练/10222.762955665588GPU秒；完整真实PhaseA/B结果待本进程完成。普通resume不能解决已花完的round；没有自动扩额、新建CAM或重置候选。审计见round_starvation_audit_20261013.json、planner_round_starvation_live.json及quota_recovery_review.json。

该原生run随后exit0，宿主确认PID1089774退出；Planner block与费用保持，API仍55。真实PhaseA/B未发布：native_closed:11的原两次维护额度已由cd669a69ae5d+d64fdf824512 summary用完。service仅以本次outcome.calls0推导remaining2，产生两次local preparation，但原adapter在HTTP前准确拒绝；新增pipeline failed记录及attempt保留，绝不当作两次API或收费。PhaseB revision/baseline仍0，WorkingSnapshot2/22processed/13abstracted不变。

这项remaining估计误差作为调度诊断记录，未为零费用local失败追加复杂paid-only架构，未关闭既有零API回复采用能力。真正推进仍受原规划轮6/6、闭组维护轮2/2以及候选2/2约束，不能通过普通重启或伪造新turn解除。剩余总预算245API、74训练、77.160344GPU小时仍存在，局部额度不是全局预算耗尽。

本轮收尾保护审计通过：原不可变672文件、旧phase projection两条精确内容、22日志前缀、29冻结科学源全一致（mismatch0）；新路径602、实际有追加的日志8，完整分类见artifact_audit_after_resume.json。26批XML按重复/旧用例改名去重535节点通过，非整仓单批结果；29控制源主项目/runtime完全match。当前已无活动训练、worker或未知API，未commit/push。续研建议的字段及范围保存在quota_recovery_review.json，仅供审查，尚无用户批准或实际应用。
