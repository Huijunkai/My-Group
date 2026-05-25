# 教务系统后端服务

教务系统数据同步与推送服务，为 HarmonyOS 原生应用 [青序](https://github.com/yiqi-jing/qinxu) 提供后端 API 支持。

## 项目概述

本服务是一个 Node.js 后端应用，主要功能包括：

- **教务数据同步**：从强智教务系统抓取学生课表、成绩、考试安排等数据
- **数据加密传输**：使用 AES-256-CBC 加密敏感数据，确保传输安全
- **实时推送通知**：集成华为推送服务，支持成绩发布、考试安排、电费提醒等通知
- **电费查询监控**：支持南宁/桂林校区宿舍电费查询与低余额提醒
- **空教室查询**：提供校区空教室查询服务
- **公告通知**：抓取教务处公告并推送关键通知

## 技术栈

| 技术        | 说明                          |
| --------- | --------------------------- |
| Node.js   | 运行环境 (v18+)                 |
| Express   | Web 框架                      |
| Sequelize | ORM 数据库框架                   |
| MySQL     | 数据库 (华为云 MariaDB)           |
| Cheerio   | HTML 解析库                    |
| Axios     | HTTP 请求库                    |
| 华为推送      | 消息推送服务                      |

## 项目结构

```
f:\jw-backend/
├── server.js                 # 应用入口，路由定义
├── package.json              # 项目依赖配置
│
├── src/
│   ├── index.js              # Express 应用配置
│   │
│   ├── api/                  # API 接口层
│   │   ├── auth.js           # 登录认证 API
│   │   ├── student.js        # 学生数据获取 (课表/成绩/考试/计划)
│   │   ├── announcement.js   # 教务公告抓取
│   │   ├── emptyroom.js      # 空教室查询
│   │   ├── electricity.js    # 电费查询 API
│   │   └── water.js          # 校园打水服务
│   │
│   ├── parser/               # HTML 解析层
│   │   └── index.js          # 教务系统页面解析器
│   │
│   ├── xyyxt/                # 校园一信通集成
│   │   ├── index.js          # 一信通主入口
│   │   ├── auth.js           # 一信通认证
│   │   ├── guilinElec.js     # 桂林校区电费查询
│   │   └── constants.js      # 常量定义
│   │
│   ├── services/             # 业务服务层
│   │   ├── pushService.js    # 华为推送服务封装
│   │   ├── realtimePush.js   # 实时推送逻辑
│   │   ├── courseReminderPush.js  # 课程提醒推送服务
│   │   ├── electricityMonitor.js  # 电费监控服务
│   │   ├── notificationMonitor.js # 通知监控服务
│   │   └── timetableSync.js  # 课表自动同步服务
│   │
│   ├── db/                   # 数据库层
│   │   ├── index.js          # 数据库连接配置
│   │   ├── sync.js           # 数据同步函数
│   │   └── models/
│   │       └── index.js      # Sequelize 模型定义
│   │
│   └── utils/                # 工具层
│       ├── constants.js      # 常量定义
│       ├── encryption.js     # AES 加密工具
│       └── request.js        # HTTP 请求封装
│
├── tests/                    # 测试文件
│   ├── test_encryption.js    # 加密测试
│   ├── test_parse_timetable.js # 课表解析测试
│   └── ...
│
└── docs/
    ├── PROJECT_PLAN.md       # 项目规划文档
    └── ADMIN_DASHBOARD.md    # 管理后台文档
```

## 核心模块说明

### 1. 数据获取流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据获取架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │  server.js  │───▶│  api/*.js   │───▶│  parser     │───▶│  encryption │  │
│   │  (路由入口)  │    │ (数据获取)   │    │ (HTML解析)   │    │ (数据加密)   │  │
│   └─────────────┘    └──────┬──────┘    └─────────────┘    └─────────────┘  │
│         │                   │                                              │
│         │                   ▼                                              │
│         │            ┌─────────────┐                                       │
│         │            │   request   │                                       │
│         │            │ (HTTP请求)   │                                       │
│         │            └──────┬──────┘                                       │
│         │                   │                                              │
│         ▼                   ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    强智教务系统                                       │  │
│   │              http://qzjw.bwgl.cn/gllgdxbwglxy_jsxsd                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. API 接口

| 接口                          | 方法       | 功能        | 文件              |
| --------------------------- | -------- | --------- | --------------- |
| `/api/sync`                 | POST     | 同步学生所有数据  | server.js       |
| `/api/login`                | POST     | 登录教务系统    | auth.js         |
| `/api/semester/latest`      | GET      | 获取最新学期    | server.js       |
| `/api/announcements`        | GET      | 获取公告列表    | announcement.js |
| `/api/announcements/detail` | GET      | 获取公告详情    | announcement.js |
| `/api/emptyroom/campuses`   | GET      | 获取校区列表    | emptyroom.js    |
| `/api/emptyroom/buildings`  | GET      | 获取楼栋列表    | emptyroom.js    |
| `/api/emptyroom/query`      | POST     | 查询空教室     | emptyroom.js    |
| `/api/electricity`          | GET      | 查询电费      | electricity.js  |
| `/api/electricity/settings` | GET/POST | 电费提醒设置    | electricity.js  |
| `/api/water/scan`           | POST     | 打水服务扫码    | water.js        |
| `/api/push/register`        | POST     | 注册推送Token | server.js       |
| `/api/push/unregister`      | POST     | 注销推送Token | server.js       |
| `/api/push/test`            | POST     | 测试推送功能    | server.js       |
| `/api/course-reminder/config` | POST   | 更新课程提醒配置 | server.js       |
| `/api/course-reminder/user-config` | POST | 更新用户课程提醒配置 | server.js |
| `/api/course-reminder/user-config/:studentId` | GET | 获取用户课程提醒配置 | server.js |

### 3. 数据加密

使用 AES-256-CBC 算法加密敏感数据：

```javascript
// 加密配置
const ALGORITHM = 'aes-256-cbc';
const KEY_LENGTH = 32;    // 密钥长度 32 字节
const IV_LENGTH = 16;     // 初始向量 16 字节

// 密钥派生
key = crypto.scryptSync(ENCRYPTION_KEY, 'salt', KEY_LENGTH);

// 加密格式
encryptedData = 'ENC:' + cipher.update(data, 'utf8', 'base64') + cipher.final('base64');
```

**加密字段：**

| 数据类型 | 加密字段                                        |
| ---- | ------------------------------------------- |
| 学生信息 | name, gender, className, major, college     |
| 课程   | name, teacher, location, weeks, courseType  |
| 成绩   | courseName, score, credit, gradePoint       |
| 考试   | courseName, location, seatNumber            |
| 计划   | courseName, teachingUnit, credit            |
| 进度   | category, requiredCredits, completedCredits |

### 4. 数据库模型

| 模型                  | 说明      | 主要字段                                                         |
| ------------------- | ------- | ------------------------------------------------------------ |
| Student             | 学生信息    | studentId, name, className, major, college, semesterStartDate  |
| Course              | 课程表     | studentId, semester, name, dayOfWeek, week, period, location, weeks |
| Grade               | 成绩记录    | studentId, semester, courseCode, courseName, score, credit   |
| Exam                | 考试安排    | studentId, courseName, examTime, location, seatNumber        |
| Plan                | 培养计划    | studentId, semester, courseCode, courseName, credit          |
| Progress            | 学分进度    | studentId, category, requiredCredits, completedCredits       |
| UserPushToken       | 推送Token | studentId, pushToken, isActive                               |
| ElectricityReminder | 电费提醒    | studentId, threshold, roomId, enabled                        |
| CourseReminderConfig | 课程提醒配置 | studentId, semesterStartDate, currentWeek, beforeClassMinutes, tomorrowHour, tomorrowMinute, enabled |

### 5. 推送服务

#### 5.1 推送类型

| 类型                   | 触发条件      | 说明              | 数据字段                                  |
| -------------------- | --------- | --------------- | ----------------------------------- |
| `new_grade`          | 新成绩发布     | 成绩同步时检测到新成绩     | courseName, score, credit, semester |
| `new_exam`           | 新考试安排     | 考试安排同步时检测到新考试   | courseName, examTime, location      |
| `exam_reminder`      | 考试提醒      | 考试前24小时提醒       | courseName, examTime, location      |
| `course_change`      | 课程变动      | 课表变更检测          | changeType, courseName              |
| `electricity_reminder` | 电费不足      | 余额低于设定阈值        | balance, threshold                  |
| `announcement`       | 公告通知      | 关键词匹配的教务公告      | title, keyword, url                 |

#### 5.2 课程提醒服务

课程提醒服务支持两种提醒方式：

- **课前提醒**：课程开始前指定分钟数提醒（默认15分钟）
- **明日提醒**：每天指定时间推送明日课程列表（默认21:00）

服务会自动处理跨周情况（周日晚上推送周一课程时自动切换到下一周）。

#### 5.3 监控服务

| 监控服务 | 检查间隔 | 说明 |
| ---- | ---- | ---- |
| 公告监控 | 每10分钟 | 监控教务处公告，匹配关键词推送 |
| 数据监控 | 每30分钟 | 检测成绩、考试、课表变化 |
| 电费监控 | 每小时 | 检查宿舍电费余额，低于阈值提醒 |
| 课表同步 | 每30分钟 | 自动同步用户课表 |

## 安装与运行

### 环境要求

- Node.js >= 18.0.0
- npm >= 9.0.0
- MySQL/MariaDB（或使用 SQLite 进行开发）

### 安装依赖

```bash
cd f:\jw-backend
npm install
```

### 配置环境变量

创建 `.env` 文件：

```env
# 服务端口
PORT=3000

# 数据库配置
DB_NAME=app_db
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_PORT=3306

# 加密密钥 (生产环境请更换)
ENCRYPTION_KEY=NNLG-HarmonyOS-2024-Secret-Key!!
ENCRYPTION_IV=NNLG-InitVector16

# 华为推送配置
HUAWEI_PROJECT_ID=your_project_id
HUAWEI_CLIENT_ID=your_client_id
HUAWEI_CLIENT_SECRET=your_client_secret

# 课程提醒配置
COURSE_REMINDER_ENABLED=true
```

### 启动服务

```bash
# 开发模式
npm start
```

### 运行测试

```bash
# 运行特定测试
node tests/test_encryption.js
node tests/test_parse_timetable.js
```

## API 使用示例

### 同步学生数据

```bash
curl -X POST http://localhost:3000/api/sync \
  -H "Content-Type: application/json" \
  -d '{
    "username": "学号",
    "password": "密码",
    "semester": "2024-2025-1"
  }'
```

### 查询空教室

```bash
curl -X POST http://localhost:3000/api/emptyroom/query \
  -H "Content-Type: application/json" \
  -d '{
    "semester": "2024-2025-1",
    "campus": "01",
    "building": "J01",
    "weekStart": 1,
    "weekEnd": 1,
    "periodStart": "0102",
    "periodEnd": "0304"
  }'
```

### 查询电费

```bash
curl "http://localhost:3000/api/electricity?username=学号&roomId=H4320&areaId=glxq&buildingId=4320"
```

## 安全特性

| 特性   | 说明                   |
| ---- | -------------------- |
| 传输加密 | AES-256-CBC 加密所有敏感数据 |
| 密钥派生 | 使用 scrypt 从密码派生密钥    |
| 数据标识 | 加密数据以 `ENC:` 前缀标识    |
| 密钥分发 | 通过独立 API 分发密钥，支持动态更新 |
| 匿名推送 | 使用哈希生成的匿名ID进行推送      |

## 相关项目

- [QinXu HarmonyOS App](https://github.com/yiqi-jing/qinxu) - HarmonyOS 原生应用前端

## 许可证

MIT License