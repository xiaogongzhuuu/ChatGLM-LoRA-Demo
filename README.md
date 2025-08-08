# ChatGLM-LoRA-Demo
ChatGLM 微调示例

## 项目介绍

这是一个基于ChatGLM3-6B大语言模型的微调项目，使用LoRA技术进行高效训练。项目包含模型训练和Web界面展示两部分。

## 主要功能

- **模型微调**: 使用自定义中文数据集训练ChatGLM模型
- **Web演示**: 提供简洁的对话界面测试模型效果
- **低成本训练**: LoRA技术大幅降低训练资源需求

## 技术栈

- **深度学习框架**: PyTorch + Transformers
- **微调技术**: LoRA (Low-Rank Adaptation) 
- **Web框架**: Streamlit
- **基础模型**: ChatGLM3-6B

## 使用方法

### 1. 安装环境
```bash
pip install -r requirements.txt
```

### 2. 开始训练
```bash
python scripts/train.py
```

### 3. 启动演示
```bash
cd app
streamlit run app.py
```

## 项目结构

```
├── app/app.py           # Web演示界面
├── data/train.json      # 训练数据(100+条中文样本)
├── scripts/train.py     # 训练脚本
└── requirements.txt     # 依赖包
```

## 数据集示例

训练数据包含多种任务类型：
- 中英文翻译
- 问答对话  
- 文本分析
- 数学计算
- 内容生成

## 项目亮点

✅ **高效训练** - LoRA技术减少90%训练参数
✅ **实用性强** - 包含丰富中文指令数据
✅ **易于部署** - 一键启动Web演示界面
✅ **成本友好** - 消费级GPU即可运行

---

**开源协议**: MIT License