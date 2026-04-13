# 训练结果前端看板

这个目录是独立前端项目，用于展示 `output/*.out` 中的训练结果，以及指定用户在目标域上的 Top10 推荐结果（跨域前/跨域后对比）。

## 功能
- 选择 `GNN 模型`
- 选择 `源域数量 (4/8/16)`
- 选择某一次训练记录并展示：
  - 元信息（dataset / dp / eps / seed / 文件名等）
  - Final 与 Best 指标卡片
  - HR@10 / NDCG@10 训练曲线
  - 各域最新轮次指标表格
- 输入 `内部用户索引`，展示目标域候选集上的：
  - 跨域前 Top10 物品
  - 跨域后 Top10 物品

## 使用步骤
1. 生成训练结果数据：
   - `python ./training-results-web/scripts/build_results_data.py`
2. 生成推荐对比数据（依赖 KT checkpoint）：
   - `.\.venv\Scripts\python.exe ./training-results-web/scripts/build_recommendation_data.py`
3. 启动静态服务（任选其一）:
   - `python -m http.server 8080`
   - 或使用你常用的前端静态服务
4. 打开页面：
   - `http://localhost:8080/training-results-web/`

## 数据来源
- 训练日志：`output/**/*.out` -> `training-results-web/data/results.json`
- 推荐快照：`checkpoints/kt_*/` + `data/*domains/implicit.json` -> `training-results-web/data/recommendations.json`
