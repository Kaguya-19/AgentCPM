# AgentDock

<p align="center">
  【中文 | <a href="README.md">English</a>】
</p>

<p align="center">
  <strong>Dynamic MCP Container Orchestration Platform</strong>
</p>

<p align="center">
  一个基于 Docker 的容器编排系统，用于部署与管理<br>
  MCP（Model Context Protocol）服务，并支持动态生命周期管理。
</p>

---

## ✨ 核心特性（Features）

- 🐳 **动态容器管理（Dynamic Container Management）**  
  支持 Docker 容器的自动创建、启动、停止与回收（remove），实现 MCP 服务的弹性调度。

- 📊 **资源监控（Resource Monitoring）**  
  提供 CPU、内存与磁盘等系统资源的实时监测能力。

- 🔄 **健康检查（Health Checks）**  
  支持容器健康状态自动检测，并在异常情况下触发自动重启机制。

- 🌐 **MCP 路由（MCP Routing）**  
  面向 Streamable-HTTP 的 MCP 协议请求转发与路由机制。

- 📈 **Web 控制台（Web Dashboard）**  
  提供可视化的容器管理与监控界面，便于运维与调试。

---

## 🏗️ 系统架构（Architecture）

```
AgentDock/
├── master/                     # 管理服务（Manager Service）
│   ├── main.py                 # FastAPI 主入口
│   ├── config.py               # 配置管理模块
│   ├── node.py                 # 节点管理相关路由
│   ├── config.yml              # 默认配置文件
│   ├── dockerfile              # Docker 构建文件
│   └── templates/              # Web 前端模板
├── node/                       # 基础 Node 镜像
├── agentdock-node-full/        # 全功能 MCP Node
├── agentdock-node-explore/     # Explore 型 MCP Node（搜索与分析）
├── docker-compose.yml          # Docker Compose 编排配置
└── .env.example                # 环境变量模板
```

---

## 🚀 快速启动（Quick Start）

### 1. 环境配置（Configure Environment）

```bash
cp .env.example .env
# 根据需要编辑 .env 文件中的 MongoDB 相关配置
```

### 2. 启动服务（Start Services）

```bash
docker compose up -d
```

### 3. 访问管理界面（Access Dashboard）

```
http://localhost:8080
```

---

## 📦 服务组件（Services）

| 服务名称 | 功能描述 | 端口 |
|---------|----------|------|
| `agentdock-manager` | 主控编排与管理控制台 | 8080 |
| `agentdock-mongodb` | 节点与状态持久化数据库 | 27017 |
| `agentdock-node-full` | 全功能 MCP Server | 8004, 8092 |
| `agentdock-node-explore` | Explore MCP Server（搜索与分析） | 8014, 8102 |

---

## ⚙️ 系统配置（Configuration）

### 环境变量（Environment Variables）

| 变量名 | 描述 | 是否必需 |
|------|------|---------|
| `MONGODB_USERNAME` | MongoDB 用户名 | ✅ |
| `MONGODB_PASSWORD` | MongoDB 密码 | ✅ |
| `JINA_API_KEY` | Jina Reader API Key | ❌ |
| `GOOGLE_SERP_API_KEY` | Google SERP API Key | ❌ |

### 资源限制（Resource Limits）

- **agentdock-manager**：2 CPU / 4GB 内存  
- **agentdock-mongodb**：2 CPU / 6GB 内存  
- **agentdock-node-full**：8 CPU / 32GB 内存  
- **agentdock-node-search**：4 CPU / 16GB 内存  

---

## 📄 许可证（License）

本仓库代码基于 Apache-2.0 协议开源发布。
