# AI-EDU

## 文档

 基于 kotaemon 改进的的智能教育考试系统

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)

基于 kotaemon 的智能教育考试系统，集成了基于 AI Agent 的智能题目生成和个性化问答功能。

## 特性亮点

- 基于 AI Agent 的智能题目生成
- 智能错题分析和个性化学习建议
- 自适应学习路径规划
- 基于历史错题的智能对话
- 灵活的题目修改和优化建议

## 系统架构

### Agent 系统架构
![技术架构](docs/images/KGQuizMaster_structure.png)

## 快速开始

### 环境要求
- Python 3.10+
- Docker (可选)

### 安装步骤

1. 克隆仓库
```bash
git clone [repository-url]
cd ai_edu_exam
```

2. 安装依赖
```bash
pip install -r doc_env_reqs.txt
```

3. 启动应用
```bash
python app.py
```


## 使用指南

### 考试场景
![学习场景1](docs/images/learn1.png)
![学习场景2](docs/images/learn2.png)

### 题目生成
![题目生成](docs/images/quiz_generate.png)

### 题目修改
![社区互动](docs/images/community.png)

### 修改建议
![问卷调查](docs/images/surveys.png)

### 错题分析系统
![推理模式](docs/images/ReWOO.png)

## 技术栈

- 框架：kotaemon
- AI 引擎：LLM, Embedding
- 数据存储：JSON, Chroma
- 容器化：Docker
