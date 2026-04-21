# 教务系统后端服务

教务系统数据同步与推送服务，为 HarmonyOS 原生应用 [青序 ](https://github.com/your-repo/qinxu)提供后端 API 支持。

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
| SQLite    | 默认数据库 (支持 MySQL/PostgreSQL) |
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
│   │   ├── electricityMonitor.js  # 电费监控服务
│   │   └── notificationMonitor.js # 通知监控服务
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
    └── PROJECT_PLAN.md       # 项目规划文档
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
| `/api/encryption/key`       | GET      | 获取加密密钥    | server.js       |

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
| Student             | 学生信息    | studentId, name, className, major, college                   |
| Course              | 课程表     | studentId, semester, name, dayOfWeek, week, period, location |
| Grade               | 成绩记录    | studentId, semester, courseCode, courseName, score, credit   |
| Exam                | 考试安排    | studentId, courseName, examTime, location, seatNumber        |
| Plan                | 培养计划    | studentId, semester, courseCode, courseName, credit          |
| Progress            | 学分进度    | studentId, category, requiredCredits, completedCredits       |
| UserPushToken       | 推送Token | studentId, pushToken, isActive                               |
| ElectricityReminder | 电费提醒    | studentId, threshold, roomId, enabled                        |

### 5. 推送服务

#### 5.1 推送服务架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              推送服务架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐                                                            │
│   │  前端应用   │                                                            │
│   │ (HarmonyOS) │                                                            │
│   └──────┬──────┘                                                            │
│          │ 1. 获取Push Token (华为SDK)                                        │
│          │                                                                   │
│          ├──────────────────────────────┐                                    │
│          │                              │                                    │
│          ▼                              ▼                                    │
│   ┌─────────────┐                 ┌──────────────┐                          │
│   │  后端API    │                 │ 华为推送服务  │                          │
│   │ /api/push/  │                 │              │                          │
│   │  register   │                 └──────┬───────┘                          │
│   └──────┬──────┘                        │                                  │
│          │ 2. 存储Token                   │                                  │
│          ▼                               │                                  │
│   ┌─────────────┐                        │                                  │
│   │   数据库    │                        │                                  │
│   │UserPushToken│                        │                                  │
│   └─────────────┘                        │                                  │
│                                           │                                  │
│          ┌────────────────────────────────┘                                  │
│          │ 3. 发送推送消息                                                   │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │  │
│   │  │ pushService  │───▶│ realtimePush │───▶│   华为API    │          │  │
│   │  │  (推送封装)   │    │  (实时推送)   │    │              │          │  │
│   │  └──────────────┘    └──────────────┘    └──────────────┘          │  │
│   │                                                                     │  │
│   │  ┌──────────────────────┐    ┌──────────────────────┐             │  │
│   │  │ notificationMonitor  │    │ electricityMonitor   │             │  │
│   │  │    (定时监控服务)      │    │   (电费监控服务)      │             │  │
│   │  └──────────────────────┘    └──────────────────────┘             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2 推送类型

| 类型                   | 触发条件      | 说明              | 数据字段                                  |
| -------------------- | --------- | --------------- | ----------------------------------- |
| `new_grade`          | 新成绩发布     | 成绩同步时检测到新成绩     | courseName, score, credit, semester |
| `new_exam`           | 新考试安排     | 考试安排同步时检测到新考试   | courseName, examTime, location      |
| `exam_reminder`      | 考试提醒      | 考试前24小时提醒       | courseName, examTime, location      |
| `course_change`      | 课程变动      | 课表变更检测          | changeType, courseName              |
| `electricity_reminder` | 电费不足      | 余额低于设定阈值        | balance, threshold                  |
| `announcement`       | 公告通知      | 关键词匹配的教务公告      | title, keyword, url                 |

#### 5.3 推送消息数据结构

```javascript
{
    validate_only: false,
    message: {
        android: {
            notification: {
                title: "通知标题",
                body: "通知内容",
                click_action: {
                    type: 3  // 点击打开应用
                }
            }
        },
        token: ["设备Token"],  // 单设备推送
        // 或 topic: "user_xxx",  // 主题订阅推送
        data: JSON.stringify({
            type: "new_grade",  // 通知类型
            courseName: "高等数学",
            score: "95",
            credit: "4"
        })
    }
}
```

#### 5.4 推送API接口

**注册推送Token**
```bash
POST /api/push/register
Content-Type: application/json

{
    "studentId": "学号",
    "pushToken": "华为推送Token",
    "deviceInfo": "设备信息"
}
```

**注销推送Token**
```bash
POST /api/push/unregister
Content-Type: application/json

{
    "studentId": "学号"
}
```

**测试推送**
```bash
POST /api/push/test
Content-Type: application/json

{
    "studentId": "学号",
    "type": "new_grade",
    "title": "测试通知",
    "content": "这是一条测试消息"
}
```

#### 5.5 推送服务流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          推送服务完整流程                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 初始化阶段                                                              │
│     └── pushService.initialize(context)                                  │
│         └── 配置华为推送SDK                                                 │
│                                                                          │
│  2. 注册接收器 (前端)                                                        │
│     └── pushNotificationService.registerMessageReceiver(ability)         │
│         └── 监听推送消息                                                    │
│                                                                          │
│  3. 获取Push Token (前端)                                                  │
│     └── pushService.getPushToken()                                       │
│         └── 调用华为SDK获取设备Token                                         │
│                                                                          │
│  4. 注册到后端                                                              │
│     └── POST /api/push/register { studentId, pushToken }                 │
│         └── 存储到数据库 UserPushToken表                                    │
│         └── 注册到监控服务                                                   │
│                                                                          │
│  5. 绑定用户 (可选)                                                          │
│     └── pushService.bindAppProfileId(studentId)                         │
│         └── 绑定用户ID用于主题订阅                                             │
│                                                                          │
│  6. 接收消息 (前端)                                                          │
│     └── receiveMessage() 回调                                             │
│         └── 解析消息数据                                                     │
│         └── 根据type处理不同通知类型                                          │
│                                                                          │
│  7. 显示通知 (前端)                                                          │
│     └── showLocalNotification(title, body, data)                        │
│         └── 显示系统通知                                                     │
│         └── 处理用户点击                                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 5.6 监控服务

**公告监控** (notificationMonitor.js)
- **检查间隔**：每10分钟检查一次
- **监控来源**：
  - 教务处：`https://jwc.bwgl.cn/tzgg/`
  - 文理学院：`https://wlxy.bwgl.cn/tzgg/`
- **监控关键词**：`重修`、`补考`、`体质健康测试`、`选课`、`补修`、`免修`
- **推送规则**：标题匹配关键词时推送

**数据监控** (notificationMonitor.js)
- **检查间隔**：每30分钟检查一次
- **监控内容**：
  - 成绩变化：对比课程名+学期
  - 考试变化：对比课程名+考试时间
  - 课表变化：检测新增、取消、教室变更
  - 考试提醒：考试前24小时自动提醒

**电费监控** (electricityMonitor.js)
- **检查间隔**：每小时检查一次
- **触发条件**：余额 < 设定阈值
- **支持校区**：南宁校区、桂林校区

#### 5.7 实时推送

当用户同步数据时,系统会自动检测新数据并立即推送:

```javascript
// 成绩同步时检测新成绩
const gradeResults = await syncGrades(username, grades);
if (gradeResults && gradeResults.length > 0) {
    for (const result of gradeResults) {
        if (result.success) {
            await realtimePush.notifyNewGradeRealtime(username, result.grade);
        }
    }
}

// 考试同步时检测新考试
const examResults = await syncExams(username, exams);
if (examResults && examResults.length > 0) {
    for (const result of examResults) {
        if (result.success) {
            await realtimePush.notifyNewExamRealtime(username, result.exam);
        }
    }
}
```

#### 5.8 匿名推送

为保护用户隐私,推送服务使用哈希生成的匿名ID:

```javascript
function generateAnonymousId(studentId) {
    let hash = 0;
    for (let i = 0; i < studentId.length; i++) {
        const char = studentId.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return `nnlg_${Math.abs(hash).toString(16)}`;
}
```

**示例**：
- 学号 `20210001` → 匿名ID `nnlg_a1b2c3d`
- 推送主题 `user_nnlg_a1b2c3d`

## 安装与运行

### 环境要求

- Node.js >= 18.0.0
- npm >= 9.0.0

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
DB_DIALECT=sqlite
DB_STORAGE=./data.db

# 加密密钥 (生产环境请更换)
ENCRYPTION_KEY=NNLG-HarmonyOS-2024-Secret-Key!!
ENCRYPTION_IV=NNLG-InitVector16

# 华为推送配置
HUAWEI_PROJECT_ID=your_project_id
HUAWEI_CLIENT_ID=your_client_id
HUAWEI_CLIENT_SECRET=your_client_secret
```

### 启动服务

```bash
# 开发模式
npm run dev

# 生产模式
npm start
```

### 运行测试

```bash
# 运行所有测试
npm test

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

**响应示例：**

```json
{
  "success": true,
  "data": {
    "info": { "name": "ENC:...", "className": "ENC:..." },
    "courses": [...],
    "grades": { "2024-2025-1": [...] },
    "exams": [...],
    "plans": { "2024-2025-1": [...] },
    "progress": [...]
  },
  "changes": {
    "newGrades": [...],
    "newExams": [...]
  }
}
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

## 数据同步逻辑

### 同步流程

```
POST /api/sync
    │
    ├── 1. 登录教务系统
    │       └── login(username, password) → cookies
    │
    ├── 2. 并行获取所有数据
    │       ├── getStudentInfo(cookies)
    │       ├── getTimetable(cookies, semester)
    │       ├── getGrades(cookies)
    │       ├── getExamSchedule(cookies)
    │       ├── getSemesterPlan(cookies)
    │       └── getStudyProgress(cookies)
    │
    ├── 3. 同步到数据库
    │       ├── syncStudent() → 新学生记录
    │       ├── syncCourses() → 按学期去重
    │       ├── syncGrades() → 检测新成绩 → 实时推送
    │       ├── syncExams() → 检测新考试 → 实时推送
    │       ├── syncPlans() → 培养计划
    │       └── syncProgress() → 学分进度
    │
    └── 4. 返回加密数据
            └── { info, courses, grades, exams, plans, progress }
```

### 去重策略

- **课程**：按 `(studentId, semester, name, dayOfWeek, week, period)` 去重
- **成绩**：按 `(studentId, semester, courseCode)` 去重
- **考试**：按 `(studentId, courseName, examTime)` 去重
- **计划**：按 `(studentId, semester, courseCode)` 去重
- **进度**：按 `(studentId, category)` 去重

## 监控服务

### 电费监控

- **检查间隔**：每小时检查一次
- **触发条件**：余额 < 设定阈值
- **通知方式**：华为推送
- **支持校区**：南宁校区、桂林校区
- **服务文件**：`src/services/electricityMonitor.js`

**监控流程**：
```
每小时触发
    │
    ├── 查询所有启用电费提醒的用户
    │
    ├── 遍历每个用户
    │       │
    │       ├── 查询电费余额
    │       │
    │       ├── 判断余额 < 阈值
    │       │       │
    │       │       ├── 是 → 发送推送通知
    │       │       └── 否 → 跳过
    │       │
    │       └── 更新最后检查时间
    │
    └── 记录监控日志
```

### 公告监控

- **检查间隔**：每10分钟检查一次
- **监控来源**：
  - 教务处：`https://jwc.bwgl.cn/tzgg/`
  - 文理学院：`https://wlxy.bwgl.cn/tzgg/`
- **推送规则**：标题匹配关键词时推送
- **服务文件**：`src/services/notificationMonitor.js`

**监控关键词**：
- `重修` - 重修通知
- `补考` - 补考通知
- `体质健康测试` - 体质健康测试通知
- `选课` - 选课通知
- `补修` - 补修通知
- `免修` - 免修通知

**监控流程**：
```
每10分钟触发
    │
    ├── 抓取教务处公告列表
    │
    ├── 抓取文理学院公告列表
    │
    ├── 合并公告列表
    │
    ├── 遍历每个公告
    │       │
    │       ├── 判断标题是否匹配关键词
    │       │       │
    │       │       ├── 匹配 → 推送通知给所有用户
    │       │       └── 不匹配 → 跳过
    │       │
    │       └── 记录已推送公告
    │
    └── 记录监控日志
```

### 数据监控

- **检查间隔**：每30分钟检查一次
- **服务文件**：`src/services/notificationMonitor.js`

**监控内容**：

| 监控项   | 检测方式              | 推送类型          |
| ----- | ----------------- | ------------- |
| 成绩变化  | 对比课程名+学期         | `new_grade`   |
| 考试变化  | 对比课程名+考试时间       | `new_exam`    |
| 课表变化  | 检测新增、取消、教室变更     | `course_change` |
| 考试提醒  | 考试前24小时自动提醒      | `exam_reminder` |

**监控流程**：
```
每30分钟触发
    │
    ├── 遍历所有注册推送的用户
    │
    ├── 对每个用户
    │       │
    │       ├── 登录教务系统
    │       │
    │       ├── 获取最新成绩
    │       │       │
    │       │       ├── 对比数据库中的成绩
    │       │       ├── 检测新成绩 → 推送通知
    │       │       └── 更新数据库
    │       │
    │       ├── 获取最新考试安排
    │       │       │
    │       │       ├── 对比数据库中的考试
    │       │       ├── 检测新考试 → 推送通知
    │       │       └── 更新数据库
    │       │
    │       ├── 获取最新课表
    │       │       │
    │       │       ├── 对比数据库中的课表
    │       │       ├── 检测课表变动 → 推送通知
    │       │       └── 更新数据库
    │       │
    │       └── 检查考试提醒
    │               │
    │               ├── 查找24小时内的考试
    │               └── 发送考试提醒推送
    │
    └── 记录监控日志
```

### 实时推送

当用户主动同步数据时,系统会立即检测新数据并推送:

**服务文件**：`src/services/realtimePush.js`

**推送时机**：
- 用户调用 `/api/sync` 接口时
- 检测到新成绩 → 立即推送
- 检测到新考试 → 立即推送
- 检测到课表变动 → 立即推送

**优势**：
- 无需等待定时监控
- 用户第一时间收到通知
- 减轻服务器定时任务压力

## 与前端协作

### 数据同步流程

前端 HarmonyOS 应用通过以下流程获取数据：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              前后端协作流程                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                      ┌─────────────┐                        │
│  │ HarmonyOS   │                      │   后端服务   │                        │
│  │ 前端应用     │                      │  (本项目)    │                        │
│  └──────┬──────┘                      └──────┬──────┘                        │
│         │                                    │                               │
│         │  1. POST /api/sync                 │                               │
│         │  { username, password }            │                               │
│         │ ─────────────────────────────────▶ │                               │
│         │                                    │                               │
│         │                                    │ 2. 登录教务系统                │
│         │                                    │    抓取并解析数据              │
│         │                                    │    加密敏感字段                │
│         │                                    │                               │
│         │  3. 返回加密数据                    │                               │
│         │  { info: ENC:..., courses: [...] } │                               │
│         │ ◀───────────────────────────────── │                               │
│         │                                    │                               │
│         │  4. GET /api/encryption/key        │                               │
│         │ ─────────────────────────────────▶ │                               │
│         │                                    │                               │
│         │  5. 返回密钥                        │                               │
│         │  { key: base64, iv: base64 }       │                               │
│         │ ◀───────────────────────────────── │                               │
│         │                                    │                               │
│         │  6. 解密数据并存储到本地             │                               │
│         │                                    │                               │
│         ▼                                    ▼                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 推送服务集成流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          推送服务集成流程                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  前端 (HarmonyOS)                          后端 (Node.js)                    │
│  ──────────────────                        ──────────────                    │
│                                                                              │
│  1. 初始化推送服务                                                              │
│     pushService.initialize(context)                                          │
│                                                                              │
│  2. 注册消息接收器 ✅ 新增                                                       │
│     pushService.registerMessageReceiver(this)                                │
│     └── 监听华为推送消息                                                        │
│                                                                              │
│  3. 获取Push Token                                                            │
│     const token = await pushService.getToken()                               │
│                                                                              │
│  4. 注册到后端                           ┌─────────────────────────┐         │
│     POST /api/push/register           │  存储Token到数据库        │         │
│     { studentId, token }  ──────────▶ │  注册到监控服务           │         │
│                                        └─────────────────────────┘         │
│                                                                              │
│  5. 绑定用户ID (可选)                                                           │
│     pushService.bindAppProfileId(studentId)                                  │
│                                                                              │
│  6. 接收推送消息                                                                │
│     receiveMessage(message) {                                                │
│         const data = JSON.parse(message.data)                                │
│         switch(data.type) {                                                  │
│             case 'new_grade':                                                │
│                 showGradeNotification(data)                                  │
│                 break                                                        │
│             case 'new_exam':                                                 │
│                 showExamNotification(data)                                   │
│                 break                                                        │
│             // ... 其他类型                                                    │
│         }                                                                    │
│     }                                                                        │
│                                                                              │
│  7. 显示本地通知                                                                │
│     notificationManager.publish({                                            │
│         title: message.title,                                                │
│         body: message.body,                                                  │
│         extraData: data                                                      │
│     })                                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 前端推送服务配置

**build-profile.json5**
```json5
{
    "app": {
        "products": [
            {
                "name": "default",
                "signingConfig": "default",
                "capabilities": {
                    "system": [
                        "PushKit"  // 华为推送服务
                    ]
                }
            }
        ]
    }
}
```

**EntryAbility.ets**
```typescript
import pushService from './services/PushNotificationService';

export default class EntryAbility extends UIAbility {
    onCreate(want: Want, launchParam: AbilityConstant.LaunchParam) {
        // 初始化推送服务
        pushService.initialize(this.context);
        
        // 注册消息接收器 ✅ 新增
        pushService.registerMessageReceiver(this);
    }
    
    // 接收推送消息回调
    receiveMessage(message: pushService.PushMessage) {
        const data = JSON.parse(message.data);
        this.handlePushMessage(data);
    }
    
    private handlePushMessage(data: any) {
        switch (data.type) {
            case 'new_grade':
                this.showGradeNotification(data);
                break;
            case 'new_exam':
                this.showExamNotification(data);
                break;
            case 'exam_reminder':
                this.showExamReminder(data);
                break;
            case 'course_change':
                this.showCourseChangeNotification(data);
                break;
            case 'electricity_reminder':
                this.showElectricityNotification(data);
                break;
            case 'announcement':
                this.showAnnouncementNotification(data);
                break;
        }
    }
}
```

**PushNotificationService.ets**
```typescript
import push from '@ohos.push';

export default class PushNotificationService {
    private context: Context;
    private receiver: any;
    
    async initialize(context: Context) {
        this.context = context;
        // 初始化华为推送SDK
        await push.init(context);
    }
    
    // 注册消息接收器 ✅ 新增
    registerMessageReceiver(receiver: any) {
        this.receiver = receiver;
        push.on('pushMessageReceived', (message) => {
            if (this.receiver && this.receiver.receiveMessage) {
                this.receiver.receiveMessage(message);
            }
        });
    }
    
    async getPushToken(): Promise<string> {
        const token = await push.getToken();
        return token;
    }
    
    async bindAppProfileId(userId: string) {
        await push.bindAppProfileId(userId);
    }
    
    async registerToBackend(studentId: string, token: string) {
        const response = await fetch('/api/push/register', {
            method: 'POST',
            body: JSON.stringify({
                studentId: studentId,
                pushToken: token,
                deviceInfo: 'HarmonyOS'
            })
        });
        return response.json();
    }
}
```

## 安全特性

| 特性   | 说明                   |
| ---- | -------------------- |
| 传输加密 | AES-256-CBC 加密所有敏感数据 |
| 密钥派生 | 使用 scrypt 从密码派生密钥    |
| 数据标识 | 加密数据以 `ENC:` 前缀标识    |
| 密钥分发 | 通过独立 API 分发密钥，支持动态更新 |
| 匿名推送 | 使用哈希生成的匿名ID进行推送      |

## 常见问题

### Q: 登录失败怎么办？

A: 检查以下几点：

1. 学号密码是否正确
2. 教务系统是否可访问
3. 网络连接是否正常

### Q: 推送不生效？

A: 确认：

1. **华为推送配置是否正确**
   - 检查 `.env` 文件中的 `HUAWEI_PROJECT_ID`、`HUAWEI_CLIENT_ID`、`HUAWEI_CLIENT_SECRET`
   - 确认华为开发者平台配置正确

2. **前端是否正确注册**
   - 确认调用了 `pushService.registerMessageReceiver(this)`
   - 确认获取到了 Push Token
   - 确认调用了 `/api/push/register` 接口

3. **设备权限**
   - 设备是否允许推送通知
   - HarmonyOS 是否配置了 PushKit 权限

4. **后端日志检查**
   ```bash
   # 查看推送服务日志
   grep "PushService" logs/app.log
   grep "Push token" logs/app.log
   ```

### Q: 推送Token注册失败？

A: 检查：

1. **前端获取Token失败**
   - 确认华为推送SDK初始化成功
   - 确认设备网络连接正常
   - 查看华为推送服务状态

2. **后端注册失败**
   - 检查数据库连接是否正常
   - 确认 `UserPushToken` 表是否存在
   - 查看后端错误日志

3. **测试推送Token注册**
   ```bash
   curl -X POST http://localhost:3000/api/push/register \
     -H "Content-Type: application/json" \
     -d '{
       "studentId": "测试学号",
       "pushToken": "测试Token",
       "deviceInfo": "测试设备"
     }'
   ```

### Q: 电费查询失败？

A: 检查：

1. 校园一信通账号密码是否正确
2. 房间号格式是否正确
3. 校区/楼栋ID是否匹配

### Q: 如何测试推送功能？

A: 使用测试接口：

```bash
# 测试推送
curl -X POST http://localhost:3000/api/push/test \
  -H "Content-Type: application/json" \
  -d '{
    "studentId": "学号",
    "type": "new_grade",
    "title": "测试通知",
    "content": "这是一条测试消息"
  }'
```

### Q: 推送消息格式是什么？

A: 推送消息格式：

```javascript
{
    "android": {
        "notification": {
            "title": "通知标题",
            "body": "通知内容",
            "click_action": { "type": 3 }
        }
    },
    "data": "{\"type\":\"new_grade\",\"courseName\":\"高等数学\",\"score\":\"95\"}"
}
```

前端接收后需要解析 `data` 字段获取具体信息。

### Q: 如何查看推送服务状态？

A: 查看后端日志：

```bash
# 查看推送服务启动状态
grep "Push notifications" logs/app.log

# 查看监控服务状态
grep "monitoring service" logs/app.log

# 查看推送发送记录
grep "Push sent" logs/app.log

# 查看推送错误
grep "Failed to send push" logs/app.log
```

### Q: 前端如何处理不同类型的推送？

A: 根据 `data.type` 字段处理：

```typescript
switch (data.type) {
    case 'new_grade':
        // 跳转到成绩页面
        router.push({ url: 'pages/GradePage' });
        break;
    case 'new_exam':
        // 跳转到考试页面
        router.push({ url: 'pages/ExamPage' });
        break;
    case 'exam_reminder':
        // 显示考试提醒
        showExamReminder(data);
        break;
    case 'course_change':
        // 跳转到课表页面
        router.push({ url: 'pages/TimetablePage' });
        break;
    case 'electricity_reminder':
        // 跳转到电费页面
        router.push({ url: 'pages/ElectricityPage' });
        break;
    case 'announcement':
        // 打开公告详情
        openAnnouncement(data.url);
        break;
}
```

## 许可证

MIT License

## 相关项目

- [QinXu HarmonyOS App](https://github.com/your-repo/qinxu) - HarmonyOS 原生应用前端

