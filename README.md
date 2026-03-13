# 教务系统同步服务 (jw-backend)

## 项目概述

**教务系统同步服务**是南理校园助手 (NNLG) 的后端服务，通过爬虫技术代理访问强智教务系统，为学生提供课程表、成绩、考试安排等数据的同步与缓存服务。

| 属性 | 信息 |
|------|------|
| 服务名称 | 教务系统同步服务 |
| 版本 | 1.0.0 |
| 运行环境 | Node.js 14+ |
| 开发语言 | JavaScript (ES6+) |
| Web 框架 | Express 5.x |
| 数据库 | MariaDB |
| 部署平台 | 华为云 |

---

## 核心功能

### 1. 用户认证

模拟浏览器登录教务系统，获取并维护会话状态。

- **账号密码登录**：使用学号和密码登录教务系统
- **Cookie 管理**：自动管理会话 Cookie
- **会话保活**：检测会话失效并自动重新登录

### 2. 数据同步

从教务系统抓取并同步各类教务数据。

- **学生信息**：姓名、性别、学院、专业、班级等基本信息
- **课程表**：按周次、节次解析课程安排
- **成绩记录**：按学期分组的成绩数据
- **考试安排**：考试时间、地点、座位号
- **培养计划**：学期课程规划
- **学分进度**：各类学分完成情况

### 3. 数据缓存

将抓取的数据存储到数据库，支持离线查询。

- **增量更新**：只更新变化的数据
- **成绩去重**：补考/重修成绩只保留最高分
- **课表优化**：按周次拆分存储，支持快速查询

---

## 项目结构

```
jw-backend/
├── server.js                          # 应用入口文件
│                                      # - Express 服务器配置
│                                      # - 路由定义
│                                      # - 数据库初始化
│
├── package.json                       # 项目依赖配置
│
├── src/
│   ├── index.js                       # 模块导出入口
│   │
│   ├── api/                           # API 接口层
│   │   ├── auth.js                    # 认证接口
│   │   │                              # - login() 登录教务系统
│   │   │
│   │   ├── student.js                 # 学生数据接口
│   │   │                              # - getStudentInfo() 获取学生信息
│   │   │                              # - getTimetable() 获取课表
│   │   │                              # - getGrades() 获取成绩
│   │   │                              # - getExamSchedule() 获取考试安排
│   │   │                              # - getSemesterPlan() 获取培养计划
│   │   │                              # - getStudyProgress() 获取学分进度
│   │   │
│   │   └── index.js                   # API 模块导出
│   │
│   ├── db/                            # 数据库层
│   │   ├── index.js                   # 数据库连接配置
│   │   │                              # - Sequelize 实例化
│   │   │                              # - initDatabase() 初始化数据库
│   │   │
│   │   ├── models/                    # 数据模型定义
│   │   │   └── index.js               # Sequelize 模型
│   │   │                              # - Student 学生信息表
│   │   │                              # - Course 课程表
│   │   │                              # - Grade 成绩表
│   │   │                              # - Exam 考试安排表
│   │   │                              # - Plan 培养计划表
│   │   │                              # - Progress 学分进度表
│   │   │
│   │   └── sync.js                    # 数据同步逻辑
│   │                                  # - syncStudent() 同步学生信息
│   │                                  # - syncCourses() 同步课表
│   │                                  # - syncGrades() 同步成绩
│   │                                  # - syncExams() 同步考试
│   │                                  # - syncPlans() 同步培养计划
│   │                                  # - syncProgress() 同步学分进度
│   │
│   ├── parser/                        # HTML 解析层
│   │   └── index.js                   # Cheerio 解析器
│   │                                  # - parseStudentInfo() 解析学生信息
│   │                                  # - parseTimetable() 解析课表
│   │                                  # - parseGrades() 解析成绩
│   │                                  # - parseExams() 解析考试安排
│   │                                  # - parseSemesterPlan() 解析培养计划
│   │                                  # - parseStudyProgress() 解析学分进度
│   │
│   └── utils/                         # 工具层
│       ├── constants.js               # 常量定义
│       │                              # - BASE_URL 教务系统地址
│       │                              # - DEFAULT_HEADERS 默认请求头
│       │
│       └── request.js                 # HTTP 请求封装
│                                      # - formatCookies() 格式化 Cookie
│                                      # - createInstance() 创建 Axios 实例
│
├── tests/                             # 测试文件
│   ├── cli.js                         # 命令行测试工具
│   ├── test_api.py                    # Python API 测试
│   ├── test_db.js                     # 数据库测试
│   ├── test_new.js                    # 新功能测试
│   ├── test_parse_timetable.js        # 课表解析测试
│   └── fixtures/
│       └── timetable_sample.html      # 测试用课表 HTML
│
└── docs/
    └── PROJECT_PLAN.md                # 项目规划文档
```

---

## 技术架构

### 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 运行环境 | Node.js | 14+ |
| Web 框架 | Express | 5.x |
| HTTP 客户端 | Axios | 1.x |
| HTML 解析 | Cheerio | 1.x |
| ORM | Sequelize | 6.x |
| 数据库 | MariaDB | 10.x |
| 跨域处理 | CORS | 2.x |
| 编码转换 | iconv-lite | 0.7.x |
| 加密 | crypto-js | 4.x |

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        HarmonyOS App                             │
│                      (南理校园助手 NNLG)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Node.js Backend                             │
│                      (jw-backend)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Express   │  │   Parser    │  │   Sync      │              │
│  │   Router    │──│   Cheerio   │──│   Service   │              │
│  └─────────────┘  └─────────────┘  └──────┬──────┘              │
│                                            │                     │
│  ┌─────────────┐                          │                     │
│  │   Axios     │◄─────────────────────────┘                     │
│  │   HTTP      │                                                │
│  └──────┬──────┘                                                │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    强智教务系统                                   │
│                  (qzjw.bwgl.cn)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   登录页    │  │   课表页    │  │   成绩页    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MariaDB                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Student    │  │  Course     │  │  Grade      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Exam       │  │  Plan       │  │  Progress   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## API 接口

### 基础接口

#### GET /

服务状态检查。

**响应示例：**
```json
{
  "message": "教务系统同步服务已启动",
  "status": "running"
}
```

#### GET /api/version

获取服务版本和部署信息。

**响应示例：**
```json
{
  "name": "jw-backend",
  "version": "1.0.0",
  "buildTime": "2025-01-15T10:30:00.000Z",
  "railway": {
    "environment": "production",
    "service": "jw-backend",
    "gitCommit": "abc123"
  }
}
```

---

### 同步接口

#### POST /api/sync

登录教务系统并同步所有数据到数据库。

**请求体：**
```json
{
  "username": "学号",
  "password": "密码",
  "semester": "2024-2025-2"  // 可选，指定学期
}
```

**响应示例：**
```json
{
  "success": true,
  "message": "登录成功，课表已同步，其它数据正在后台同步中",
  "student": {
    "name": "张三",
    "gender": "男",
    "college": "计算机学院",
    "major": "软件工程",
    "className": "软件2201班"
  },
  "timetableCount": 45,
  "timetableDebug": {
    "ok": true,
    "reason": "synced"
  }
}
```

**错误响应：**
```json
{
  "success": false,
  "message": "登录失败，请检查学号密码"
}
```

---

### 查询接口

#### GET /api/student/:id

从数据库获取已缓存的学生数据。

**路径参数：**
- `id`: 学号

**查询参数：**
- `semester`: 可选，筛选指定学期的数据

**响应示例：**
```json
{
  "success": true,
  "data": {
    "info": {
      "studentId": "20220001",
      "name": "张三",
      "gender": "男",
      "college": "计算机学院"
    },
    "courses": [...],
    "grades": [...],
    "exams": [...],
    "plans": [...],
    "progress": [...]
  }
}
```

#### GET /api/students

获取所有已缓存的学生列表。

**响应示例：**
```json
{
  "success": true,
  "data": [
    {
      "studentId": "20220001",
      "name": "张三",
      "lastSync": "2025-01-15T10:30:00.000Z"
    }
  ]
}
```

---

## 数据模型

### Student (学生信息表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| name | STRING | 姓名 |
| gender | STRING | 性别 |
| enrollmentYear | STRING | 入学年份 |
| className | STRING | 班级 |
| major | STRING | 专业 |
| college | STRING | 学院 |
| lastSync | DATE | 最后同步时间 |

### Course (课程表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| semester | STRING(50) | 学期 (主键) |
| name | STRING(100) | 课程名称 (主键) |
| dayOfWeek | STRING(20) | 星期 (主键) |
| week | INTEGER | 周次 (主键) |
| period | STRING(50) | 节次 (主键) |
| teacher | STRING | 教师 |
| weeks | STRING | 周次字符串 |
| location | STRING | 上课地点 |
| courseType | STRING | 课程类型 |
| raw | TEXT | 原始数据 |

### Grade (成绩表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| semester | STRING(50) | 学期 (主键) |
| courseCode | STRING(50) | 课程编号 (主键) |
| courseName | STRING | 课程名称 |
| score | STRING | 成绩 |
| credit | STRING | 学分 |
| gradePoint | STRING | 绩点 |
| courseType | STRING | 课程类型 |
| examType | STRING | 考试类型 |

### Exam (考试安排表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| courseName | STRING(100) | 课程名称 (主键) |
| examTime | STRING(50) | 考试时间 (主键) |
| location | STRING | 考试地点 |
| seatNumber | STRING | 座位号 |
| examType | STRING | 考试类型 |
| status | STRING | 状态 |

### Plan (培养计划表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| semester | STRING(50) | 学期 (主键) |
| courseCode | STRING(50) | 课程编号 (主键) |
| courseName | STRING | 课程名称 |
| teachingUnit | STRING | 开课单位 |
| credit | STRING | 学分 |
| totalHours | STRING | 总学时 |
| examType | STRING | 考核方式 |
| courseAttribute | STRING | 课程属性 |
| isExam | STRING | 是否考试 |

### Progress (学分进度表)

| 字段 | 类型 | 说明 |
|------|------|------|
| studentId | STRING(50) | 学号 (主键) |
| category | STRING(50) | 课程体系 (主键) |
| requiredCredits | STRING | 要求学分 |
| completedCredits | STRING | 已完成学分 |
| currentCredits | STRING | 在修学分 |
| remainingCredits | STRING | 剩余学分 |

---

## 核心模块详解

### 1. 登录模块 (auth.js)

模拟浏览器登录教务系统的流程：

```
1. GET /xk/LoginToXk     → 获取初始 Cookie
2. 构造登录数据           → Base64 编码用户名密码
3. POST /xk/LoginToXk    → 提交登录
4. 检查 302 跳转          → 判断登录是否成功
5. 返回 Cookie           → 用于后续请求
```

### 2. 课表解析模块 (parser.js)

课表解析是最复杂的模块，需要处理多种格式：

**解析流程：**
```
1. 加载 HTML → Cheerio 解析
2. 定位课表 → #kbtable 表格
3. 遍历单元格 → 提取课程信息
4. 解析周次 → 支持 "1-16周"、"单周"、"双周" 等格式
5. 解析节次 → 支持 "01-02节"、"05-06-07节" 等格式
6. 按周拆分 → 每周每节课一条记录
```

**周次格式支持：**
- `1-16周` → [1, 2, 3, ..., 16]
- `1,3,5周` → [1, 3, 5]
- `1-8周(单)` → [1, 3, 5, 7]
- `1-8周(双)` → [2, 4, 6, 8]
- `全部` → 所有周次

### 3. 数据同步模块 (sync.js)

**课表同步优化：**
- 按学期先清空旧数据，避免周次变更后残留
- 使用 `bulkCreate` + `updateOnDuplicate` 批量插入
- 主键去重，避免重复数据

**成绩同步优化：**
- 同一课程多次考试（补考/重修）只保留最高分
- 支持等级制成绩映射（优秀=95，良好=85 等）

---

## 环境配置

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| PORT | 服务端口 | 3000 |

### 数据库配置

修改 `src/db/index.js` 中的数据库连接配置：

```javascript
const sequelize = new Sequelize(
    'app_db',           // 数据库名
    'app_backend',      // 用户名
    'your_password',    // 密码
    {
        host: '127.0.0.1',
        port: 3306,
        dialect: 'mysql',
        // ...
    }
);
```

---

## 构建与运行

### 环境要求

- Node.js 14+
- MariaDB 10.x
- npm 或 yarn

### 安装依赖

```bash
npm install
```

### 启动服务

```bash
# 开发环境
npm start

# 或直接运行
node server.js
```

### 生产部署

**华为云部署：**
1. 购买弹性云服务器
2. 安装 Node.js 和 MariaDB
3. 配置安全组开放端口
4. 使用 PM2 守护进程

```bash
# 安装 PM2
npm install -g pm2

# 启动服务
pm2 start server.js --name jw-backend

# 开机自启
pm2 startup
pm2 save
```

---

## 开发规范

### 命名规范

- **文件名**: camelCase (如 `auth.js`, `student.js`)
- **变量/函数**: camelCase (如 `getStudentInfo`)
- **常量**: UPPER_SNAKE_CASE (如 `BASE_URL`)
- **数据库表**: PascalCase 单数形式 (如 `Student`, `Course`)

### 代码规范

- 使用 ES6+ 语法
- 使用 async/await 处理异步操作
- 函数必须有 JSDoc 注释
- 错误必须捕获并返回友好提示

### 目录规范

```
src/
├── api/        # API 接口 - 处理 HTTP 请求
├── db/         # 数据库 - 模型定义和数据操作
├── parser/     # 解析器 - HTML 解析逻辑
└── utils/      # 工具 - 通用工具函数
```

---

## 错误处理

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 400 | 参数缺失 | 检查请求体是否完整 |
| 401 | 登录失败 | 检查学号密码是否正确 |
| 404 | 数据不存在 | 先调用同步接口 |
| 500 | 服务器错误 | 查看服务器日志 |
| 503 | 数据库未就绪 | 等待数据库初始化完成 |

### 错误响应格式

```json
{
  "success": false,
  "message": "错误描述信息"
}
```

---

## 安全注意事项

1. **密码安全**
   - 数据库密码字段已移除，不再存储用户密码
   - 每次同步需要前端传入密码

2. **会话管理**
   - Cookie 仅在内存中使用，不持久化
   - 会话有效期约 30 分钟

3. **数据安全**
   - 仅缓存学生自己的数据
   - 不对外暴露敏感接口

---

## 更新日志

### v1.0.0

- 实现教务系统登录认证
- 实现课表数据抓取与解析
- 实现成绩数据抓取与解析
- 实现考试安排数据抓取
- 实现培养计划数据抓取
- 实现学分进度数据抓取
- 实现数据缓存与增量更新
- 支持指定学期查询
- 优化课表解析算法
- 添加后台异步同步机制

---

## 相关项目

- [南理校园助手 (NNLG)](../My-HarmonyOS/HarmonyOS-APP/NNLG) - HarmonyOS 前端应用

---

## 许可证

本项目仅供学习交流使用，请勿用于商业用途。
