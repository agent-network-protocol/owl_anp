# ANP示例使用说明

## OWL 项目简介

🦉 OWL (Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation) 是一个前沿的多智能体协作框架，旨在推动任务自动化的边界，构建在 [CAMEL-AI Framework](https://github.com/camel-ai/camel) 之上。OWL 通过利用动态智能体交互，实现了跨多领域更自然、高效且稳健的任务自动化。

OWL 提供了多种核心功能，包括：
- **在线搜索**：使用维基百科、谷歌搜索等，进行实时信息检索
- **多模态处理**：支持互联网或本地视频、图片、语音处理
- **浏览器操作**：借助Playwright框架开发浏览器模拟交互
- **文件解析**：word、excel、PDF、PowerPoint信息提取
- **代码执行**：编写python代码，并使用解释器运行

## ANP查询功能简介

`run_anp_example.py` 是基于 OWL 框架集成开发的一个演示脚本，专门用于展示如何使用 `ANPToolkit` 进行酒店查询和推荐。该功能是 OWL 项目的一个扩展应用，展示了如何将 OWL 的多智能体协作能力应用于特定领域的任务。

该脚本使用 Azure OpenAI 服务来处理自然语言查询，并通过 `ANPToolkit` 提供的工具函数来获取ANP信息和推荐。通过这个示例，您可以了解如何将 OWL 框架应用于实际业务场景，以及如何开发自己的领域特定工具包。

## 前提条件

在运行此脚本之前，您需要满足以下条件：

1. 安装所有必要的依赖项（可在项目根目录运行 `pip install -r requirements.txt`, 具体可参考README.md/README_zh.md）
2. 配置 Azure OpenAI API 凭据（在 `.env` 文件中设置）

## 环境变量配置

在项目根目录创建一个 `.env` （可参考 `.env_template`）文件，并设置以下环境变量：

```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT=your_deployment
AZURE_OPENAI_MODEL=your_model_name
```

## 使用方法

### 基本用法

直接运行脚本即可启动ANP查询示例：

```bash
python run_anp_example.py
```

默认情况下，脚本会使用预设的查询："我需要预订杭州的一个酒店：{formatted_date}，1天的酒店，经纬度（120.026208, 30.279212）。帮我找一个酒店，然后帮我选个房间，我希望大一点，有窗户的。最后告诉我你选择的详细信息"

### 自定义查询

如果您想使用自定义查询，可以修改脚本中的 `question` 变量：

```python
# 示例问题
question = "我需要预订上海的一个酒店：2025年4月15日，2天的酒店，靠近外滩。我需要有免费早餐的房间。"

society = construct_society(question)
answer, chat_history, token_count = run_society(society)
```

## 工作原理

1. 脚本首先加载环境变量和必要的依赖项
2. 创建用户和助手模型实例（使用 Azure OpenAI）
3. 初始化 `ANPToolkit` 并获取其工具函数
4. 构建一个 `OwlRolePlaying` 社会，包含用户和助手角色
5. 运行社会模拟，处理ANP查询请求
6. 输出查询结果

## 核心组件

- **ANPToolkit**: 提供ANP查询和推荐功能的工具包
- **OwlRolePlaying**: 模拟用户和助手之间的对话
- **ModelFactory**: 创建和配置 AI 模型实例

## 示例输出

成功运行后，脚本将输出类似以下内容的酒店推荐：

```
Answer: 我已经为您选择了一家位于杭州的酒店，详情如下：

选择的酒店：杭州菲住布渴酒店
- 地址：文一西路969号乐淘城8号楼
- 星级：4星（高档型）
- 评分：4.7/5
- 距离您指定位置：440米
- 价格：起价¥599/晚

推荐房间：豪华大床房
- 床型：1张2米大床
- 面积：28平方米
- 含早餐
- 免费WiFi
- 空调、电视、独立卫浴等标准设施

这家ANP位于杭州未来科技城区域，靠近您提供的经纬度位置，步行仅需5分钟。ANP环境优雅，服务设施完善，包括共享办公空间、叫醒服务和洗衣服务等。房间宽敞舒适，非常适合商务或休闲旅行。

您可以通过ANP官网或主流订房平台预订这家ANP。如需更多信息或其他选择，请告知我。
```

## 故障排除

如果遇到问题，请检查：

1. 环境变量是否正确设置
2. 网络连接是否正常
3. Azure OpenAI 服务是否可用
4. 日志输出中是否有错误信息

## 贡献

如果您想为这个项目做出贡献，请随时提交 Pull Request 或创建 Issue。
