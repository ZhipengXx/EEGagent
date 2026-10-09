# Tasks

## 1. 身份绑定

- [x] 1.1 LLM 账本行带 candidate、attempt、phase、operation 和新的 call_id。审查和实现结果文件带身份与输入哈希。跨候选的成功调用不能当成当前阶段完成。验收：`pytest tests/eeg_research/test_attempt_isolation.py -q -k "c2_review or call_id or campaign"`。

## 2. 按结果文件恢复

- [x] 2.1 没有匹配 `review.json` 时只审查当前候选。匹配文件只提交一次。哈希不匹配则重审。job 的 candidate 不一致则阻塞。预算耗尽停在审查阶段。验收：`pytest tests/eeg_research/test_attempt_isolation.py -q -k "review or job or budget or hash or tail"`。

## 3. 回归

- [x] 3.1 跑 `test_attempt_isolation.py`、`test_crash_recovery_followups.py`、`test_worker_failure_recovery.py`、`test_execution_protocol.py`、`test_v1_8.py`。不得请求真实 DeepSeek、GPU 或训练进程。
