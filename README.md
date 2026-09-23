# 晚风 Agent

连续对话的本人原型 agent。谁跟**晚风**聊，等于跟你聊；**早安**是你的助理，不冒充你。
一次回一句，不让你从三条里挑。发送权在你手里，系统不代发微信。

「像你」靠三样东西：本机记忆（人设卡 / 关系薄卡 / 事实）、说话方式三档
（`data/style_profile.json`）、生成走 DeepSeek。

## 五分钟跑通（不需要 API Key）

```powershell
python cli.py init
python cli.py ingest examples/sample_chat.csv --self-name 我
python cli.py stats
python cli.py --offline chat "在吗，晚上一起吃饭吗"
python cli.py web
```

接上真实模型：把 `.env.example` 复制成 `.env`，填 `DEEPSEEK_API_KEY`，
然后 `python cli.py doctor` 看自检结果，去掉 `--offline` 即可。
也可以双击 `启动晚风.bat`。

## 常用命令

| 命令 | 作用 |
|---|---|
| `web` | 网页：晚风 / 早安连续对话（可以直接粘图、拖图、选图） |
| `chat "对方的消息"` | 命令行聊一句（`--mode persona\|assistant`，`--contact 谢总`，`--trace` 看工具，`--no-tools` 关掉循环） |
| `consolidate` | 把近圈固化成 `data/people/*.md`（`--check` 只检查） |
| `people` | 查看固化后的「我眼里的他」 |
| `init` | 建知识库（SQLite） |
| `ingest <文件或目录>` | 导入聊天记录（CSV，列名自动识别） |
| `extract --limit 40` | 调用模型抽取事实与偏好 |
| `persona` / `fact` / `pref` | 人设卡、手工事实、偏好 |
| `rescan-scenes` | 按 `data/contact_scenes.json` 重刷场景标签 |
| `health` / `doctor` / `stats` / `cost` | 体检、自检、统计、花费 |

## 目录

```
cli.py                命令行入口
启动晚风.bat          一键启动网页
humanagent/
  agent.py            晚风 / 早安内核：认人 + 工具循环 + 拼提示词
  consolidate.py      把近圈聊天固化成「我眼里的他」
  server.py           本地网页与 JSON 接口
  config.py           配置与密钥（只读环境变量 / .env）
  llm.py              模型网关（OpenAI 兼容，含工具调用与离线桩）
  kb.py               事实抽取、关系薄卡、人物查询、说话方式档
  db.py               SQLite 表结构（对话 / 事实 / 偏好 / 人设卡）
  ingest.py           记录导入与场景匹配
  healthcheck.py      数据体检
web/index.html        网页界面
data/
  kb.sqlite3          知识库
  contact_scenes.json 私聊联系人 → 关系大类（576 人）
  group_scenes.json   群 → 关系大类
  relations.json      近圈薄卡：这个人是谁 / 怎么说 / 还联系吗
  people/             固化后的「我眼里的他」（consolidate 生成）
  style_profile.json  说话方式三档 + 真实原话样例
  联系人分类.xlsx      上面两张分类表的来源
```

## 数据与隐私

真实聊天记录只放在本机 `data/` 下。默认只有需要生成的那一小段上下文会发给模型。
不自动发送、不接非官方发消息通道。
