# 训练结果前端看板

这个目录是独立前端项目，用于展示 `output/*.out` 中的训练结果。

## 功能
- 选择 `GNN 模型`
- 选择 `源域数量 (4/8/16)`
- 选择某一次训练记录并展示：
  - 元信息（dataset / dp / eps / seed / 文件名等）
  - Final 与 Best 指标卡片
  - HR@10 / NDCG@10 训练曲线
  - 各域最新轮次指标表格

## 使用步骤
1. 生成数据：
   - `python ./training-results-web/scripts/build_results_data.py`
2. 启动静态服务（任选其一）:
   - `python -m http.server 8080`
   - 或使用你常用的前端静态服务
3. 打开页面：
   - `http://localhost:8080/training-results-web/`

## 数据来源
- 扫描目录：`output/**/*.out`
- 输出文件：`training-results-web/data/results.json`
