# 开发者管理后台需求文档

## 1. 项目概述

### 1.1 背景
针对「勤学」HarmonyOS校园助手应用，为开发者提供一个可视化的管理后台，用于监控软件运行状态、查看使用数据、管理用户和系统配置。

### 1.2 目标用户
- 应用开发者（本人）
- 系统管理员

### 1.3 技术栈建议

#### 1.3.1 技术选型表
| 层级 | 技术选择 | 版本 | 说明 |
|------|----------|------|------|
| 前端框架 | Vue 3 | ^3.4.x | 渐进式JavaScript框架 |
| UI组件库 | Element Plus | ^2.5.x | Vue 3管理后台UI组件库 |
| 状态管理 | Pinia | ^2.1.x | Vue 3官方状态管理 |
| 路由 | Vue Router | ^4.2.x | Vue 3路由管理 |
| HTTP请求 | Axios | ^1.6.x | 复用项目现有依赖 |
| 图表库 | ECharts | ^5.5.x | 数据可视化图表 |
| 构建工具 | Vite | ^5.x | 快速构建工具 |
| 后端 | Node.js + Express | 复用现有 | 扩展现有API |
| 数据库 | MariaDB | 复用现有 | 复用现有数据库 |
| 认证 | JWT | 复用现有 | 复用现有认证中间件 |

#### 1.3.2 前端依赖清单
```json
{
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.2.0",
    "pinia": "^2.1.0",
    "element-plus": "^2.5.0",
    "axios": "^1.6.0",
    "echarts": "^5.5.0",
    "dayjs": "^1.11.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@vitejs/plugin-vue": "^5.0.0",
    "sass": "^1.69.0"
  }
}
```

#### 1.3.3 项目目录结构
```
admin-dashboard/
├── src/
│   ├── api/                 # API请求模块
│   │   ├── auth.js          # 认证接口
│   │   ├── dashboard.js     # 仪表盘接口
│   │   ├── user.js          # 用户管理接口
│   │   └── system.js        # 系统监控接口
│   ├── components/          # 公共组件
│   │   ├── Layout/          # 布局组件
│   │   ├── Charts/          # 图表组件
│   │   └── Common/          # 通用组件
│   ├── views/               # 页面视图
│   │   ├── Dashboard/       # 仪表盘
│   │   ├── User/            # 用户管理
│   │   ├── Data/            # 数据统计
│   │   ├── Monitor/         # 系统监控
│   │   ├── Api/             # API管理
│   │   ├── Push/            # 推送管理
│   │   └── Settings/        # 系统设置
│   ├── stores/              # Pinia状态管理
│   ├── router/              # 路由配置
│   ├── utils/               # 工具函数
│   └── styles/              # 全局样式
├── public/                  # 静态资源
├── .env                     # 环境变量
└── vite.config.js           # Vite配置
```

---

## 2. 功能模块设计

### 2.1 仪表盘（Dashboard）

#### 2.1.1 核心指标卡片
```
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  总用户数   │ │  今日活跃   │ │  数据请求   │ │  推送成功   │
│    156      │ │    42       │ │   1,234     │ │    98.5%    │
│  ↑ 12%      │ │  ↑ 8%       │ │  ↑ 23%      │ │  ↑ 0.5%     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

| 指标名称 | 数据来源 | 计算方式 |
|----------|----------|----------|
| 总用户数 | Student表 | COUNT(*) |
| 今日活跃 | UserPushToken表 | COUNT(lastActiveAt >= 今日) |
| 数据请求 | 日志/统计表 | 今日API调用次数 |
| 推送成功率 | 推送记录 | 成功数/总数 × 100% |

#### 2.1.2 趋势图表
- **用户增长趋势**：近30天每日新增用户折线图
- **API调用趋势**：近7天每小时请求量热力图
- **功能使用分布**：各API端点调用占比饼图

### 2.2 用户管理

#### 2.2.1 用户列表
| 字段 | 说明 | 来源 |
|------|------|------|
| 学号 | 学生唯一标识 | Student.studentId |
| 姓名 | 学生姓名 | Student.name |
| 学院 | 所属学院 | Student.college |
| 专业 | 所属专业 | Student.major |
| 班级 | 班级名称 | Student.className |
| 注册时间 | 首次同步时间 | Student.lastSync |
| 推送状态 | 是否开启推送 | UserPushToken.isActive |
| 最后活跃 | 最近一次活动 | UserPushToken.lastActiveAt |

#### 2.2.2 用户详情
- 基本信息展示
- 数据统计（课程数、成绩数、考试数）
- 活跃时间线
- 推送记录

### 2.3 数据统计

#### 2.3.1 数据量概览
```
┌────────────────────────────────────────────────────────┐
│                    数据存储统计                         │
├─────────────┬─────────────┬─────────────┬─────────────┤
│  学生信息   │   课程数据   │   成绩数据   │   考试数据   │
│   156 条    │  2,340 条    │  4,680 条    │   312 条    │
│  占用 12KB  │  占用 180KB  │  占用 320KB  │  占用 25KB  │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

#### 2.3.2 数据明细
| 数据类型 | 表名 | 统计维度 |
|----------|------|----------|
| 学生信息 | Student | 按学院/专业/年级分布 |
| 课程数据 | Course | 按学期/学生分布 |
| 成绩数据 | Grade | 按学期/学生分布 |
| 考试安排 | Exam | 按学期分布 |
| 培养计划 | Plan | 按学期分布 |
| 学分进度 | Progress | 按学生分布 |
| 推送Token | UserPushToken | 活跃/非活跃分布 |
| 电费提醒 | ElectricityReminder | 开启/关闭分布 |

### 2.4 系统监控

#### 2.4.1 服务状态
```
┌─────────────────────────────────────────────────┐
│  服务名称          状态          响应时间       │
├─────────────────────────────────────────────────┤
│  主服务 (Port 3000)  ● 运行中      45ms        │
│  数据库连接          ● 正常        12ms        │
│  推送服务            ● 正常        230ms       │
│  教务系统连接        ● 正常        560ms       │
└─────────────────────────────────────────────────┘
```

#### 2.4.2 运行模式
- 当前运行模式（Mock/Production）
- 环境变量配置
- 服务版本信息

#### 2.4.3 日志查看
- 实时日志流
- 日志级别筛选（INFO/WARN/ERROR）
- 时间范围查询

### 2.5 API管理

#### 2.5.1 API调用统计
| API端点 | 调用次数 | 平均响应时间 | 错误率 |
|---------|----------|--------------|--------|
| POST /api/sync | 1,234 | 1.2s | 0.5% |
| GET /api/emptyroom | 567 | 320ms | 0.1% |
| POST /api/push/register | 89 | 45ms | 0% |
| ... | ... | ... | ... |

#### 2.5.2 API端点管理
- 端点列表与说明
- 启用/禁用控制
- 速率限制配置

### 2.6 推送管理

#### 2.6.1 推送统计
- 今日推送数量
- 推送类型分布（成绩更新/考试提醒/电费预警）
- 推送成功率趋势

#### 2.6.2 推送记录
| 时间 | 学号 | 类型 | 标题 | 状态 |
|------|------|------|------|------|
| 2026-05-17 10:30 | 202101001 | 成绩更新 | 新成绩发布 | 成功 |
| 2026-05-17 09:15 | 202101002 | 电费预警 | 电费不足 | 成功 |

### 2.7 系统设置

#### 2.7.1 基础配置
- 运行模式切换（Mock/Production）
- 教务系统地址配置
- 数据库连接配置

#### 2.7.2 推送配置
- 推送服务开关
- 推送模板管理
- 推送频率限制

#### 2.7.3 安全设置
- 管理员密码修改
- API密钥管理
- 访问白名单

---

## 3. 数据库扩展设计

### 3.1 新增统计表

#### 3.1.1 api_stats（API调用统计表）
**用途**：记录每个API端点的调用统计，用于分析系统负载和性能。

```sql
CREATE TABLE api_stats (
    id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint VARCHAR(255) NOT NULL COMMENT 'API端点路径',
    method VARCHAR(10) NOT NULL COMMENT 'HTTP方法 GET/POST/PUT/DELETE',
    request_count INT DEFAULT 0 COMMENT '请求次数',
    avg_response_time INT DEFAULT 0 COMMENT '平均响应时间(ms)',
    error_count INT DEFAULT 0 COMMENT '错误次数',
    date DATE NOT NULL COMMENT '统计日期',
    hour TINYINT COMMENT '统计小时(0-23)，按小时统计时使用',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_stat (endpoint, method, date, hour),
    INDEX idx_date (date),
    INDEX idx_endpoint (endpoint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API调用统计表';
```

**Sequelize模型定义**：
```javascript
const ApiStats = sequelize.define('ApiStats', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    endpoint: { type: DataTypes.STRING(255), allowNull: false },
    method: { type: DataTypes.STRING(10), allowNull: false },
    requestCount: { type: DataTypes.INTEGER, defaultValue: 0 },
    avgResponseTime: { type: DataTypes.INTEGER, defaultValue: 0 },
    errorCount: { type: DataTypes.INTEGER, defaultValue: 0 },
    date: { type: DataTypes.DATEONLY, allowNull: false },
    hour: { type: DataTypes.TINYINT }
}, { tableName: 'api_stats', timestamps: true });
```

#### 3.1.2 push_logs（推送日志表）
**用途**：记录所有推送消息的发送记录，用于追踪推送状态和问题排查。

```sql
CREATE TABLE push_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    student_id VARCHAR(100) NOT NULL COMMENT '目标学号',
    push_type VARCHAR(50) NOT NULL COMMENT '推送类型: grade/exam/electricity/system',
    title VARCHAR(255) COMMENT '推送标题',
    content TEXT COMMENT '推送内容',
    status ENUM('success', 'failed', 'pending') DEFAULT 'pending' COMMENT '推送状态',
    error_message TEXT COMMENT '错误信息',
    response_data JSON COMMENT '推送服务返回数据',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_student (student_id),
    INDEX idx_type (push_type),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推送日志表';
```

**Sequelize模型定义**：
```javascript
const PushLog = sequelize.define('PushLog', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100), allowNull: false },
    pushType: { type: DataTypes.STRING(50), allowNull: false },
    title: DataTypes.STRING(255),
    content: DataTypes.TEXT,
    status: { type: DataTypes.ENUM('success', 'failed', 'pending'), defaultValue: 'pending' },
    errorMessage: DataTypes.TEXT,
    responseData: DataTypes.JSON
}, { tableName: 'push_logs', timestamps: false });
```

#### 3.1.3 system_logs（系统日志表）
**用途**：记录系统运行日志，替代console.log，便于后台查看和分析。

```sql
CREATE TABLE system_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    level ENUM('DEBUG', 'INFO', 'WARN', 'ERROR') DEFAULT 'INFO' COMMENT '日志级别',
    module VARCHAR(100) COMMENT '模块名称: api/db/push/auth等',
    message TEXT NOT NULL COMMENT '日志消息',
    metadata JSON COMMENT '附加元数据',
    ip VARCHAR(45) COMMENT '请求IP',
    user_id VARCHAR(100) COMMENT '关联用户',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_level (level),
    INDEX idx_module (module),
    INDEX idx_created (created_at),
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统日志表';
```

**Sequelize模型定义**：
```javascript
const SystemLog = sequelize.define('SystemLog', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    level: { type: DataTypes.ENUM('DEBUG', 'INFO', 'WARN', 'ERROR'), defaultValue: 'INFO' },
    module: DataTypes.STRING(100),
    message: { type: DataTypes.TEXT, allowNull: false },
    metadata: DataTypes.JSON,
    ip: DataTypes.STRING(45),
    userId: DataTypes.STRING(100)
}, { tableName: 'system_logs', timestamps: false });
```

### 3.2 管理员表

#### 3.2.1 admin_users（管理员账户表）
**用途**：存储管理后台的用户信息，支持多管理员和权限分级。

```sql
CREATE TABLE admin_users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(100) UNIQUE NOT NULL COMMENT '登录用户名',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希(bcrypt)',
    nickname VARCHAR(100) COMMENT '显示昵称',
    email VARCHAR(255) COMMENT '邮箱',
    role ENUM('super_admin', 'admin', 'viewer') DEFAULT 'admin' COMMENT '角色权限',
    permissions JSON COMMENT '细粒度权限配置',
    is_active BOOLEAN DEFAULT TRUE COMMENT '账户是否启用',
    last_login DATETIME COMMENT '最后登录时间',
    last_login_ip VARCHAR(45) COMMENT '最后登录IP',
    login_count INT DEFAULT 0 COMMENT '登录次数',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理员账户表';
```

**Sequelize模型定义**：
```javascript
const AdminUser = sequelize.define('AdminUser', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    username: { type: DataTypes.STRING(100), unique: true, allowNull: false },
    passwordHash: { type: DataTypes.STRING(255), allowNull: false },
    nickname: DataTypes.STRING(100),
    email: DataTypes.STRING(255),
    role: { type: DataTypes.ENUM('super_admin', 'admin', 'viewer'), defaultValue: 'admin' },
    permissions: DataTypes.JSON,
    isActive: { type: DataTypes.BOOLEAN, defaultValue: true },
    lastLogin: DataTypes.DATE,
    lastLoginIp: DataTypes.STRING(45),
    loginCount: { type: DataTypes.INTEGER, defaultValue: 0 }
}, { tableName: 'admin_users', timestamps: true });
```

**角色权限说明**：
| 角色 | 权限范围 |
|------|----------|
| super_admin | 所有权限，包括管理员管理、系统配置 |
| admin | 数据查看、用户管理、推送管理 |
| viewer | 仅查看仪表盘和数据统计 |

### 3.3 操作审计表

#### 3.3.1 admin_audit_logs（管理员操作日志）
**用途**：记录管理员的所有操作，用于安全审计。

```sql
CREATE TABLE admin_audit_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    admin_id INT NOT NULL COMMENT '管理员ID',
    admin_username VARCHAR(100) NOT NULL COMMENT '管理员用户名',
    action VARCHAR(100) NOT NULL COMMENT '操作类型',
    target_type VARCHAR(50) COMMENT '操作对象类型',
    target_id VARCHAR(100) COMMENT '操作对象ID',
    old_value JSON COMMENT '修改前的值',
    new_value JSON COMMENT '修改后的值',
    ip VARCHAR(45) COMMENT '操作IP',
    user_agent VARCHAR(500) COMMENT '浏览器UA',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_admin (admin_id),
    INDEX idx_action (action),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理员操作审计日志';
```

### 3.4 数据库迁移脚本

```javascript
// migrations/20260517-create-admin-tables.js
module.exports = {
    up: async (queryInterface, Sequelize) => {
        await queryInterface.createTable('api_stats', { /* ... */ });
        await queryInterface.createTable('push_logs', { /* ... */ });
        await queryInterface.createTable('system_logs', { /* ... */ });
        await queryInterface.createTable('admin_users', { /* ... */ });
        await queryInterface.createTable('admin_audit_logs', { /* ... */ });
        
        // 插入默认超级管理员
        await queryInterface.bulkInsert('admin_users', [{
            username: 'admin',
            password_hash: await bcrypt.hash('admin123', 10),
            nickname: '超级管理员',
            role: 'super_admin',
            is_active: true,
            created_at: new Date(),
            updated_at: new Date()
        }]);
    },
    down: async (queryInterface) => {
        await queryInterface.dropTable('admin_audit_logs');
        await queryInterface.dropTable('admin_users');
        await queryInterface.dropTable('system_logs');
        await queryInterface.dropTable('push_logs');
        await queryInterface.dropTable('api_stats');
    }
};
```

---

## 4. API接口设计

### 4.1 认证接口

```
POST /api/admin/login
请求: { username, password }
响应: { success, token, user: { id, username, role } }
```

### 4.2 仪表盘接口

```
GET /api/admin/dashboard/stats
响应: {
    totalUsers: number,
    activeToday: number,
    apiRequests: number,
    pushSuccessRate: number,
    trends: {
        userGrowth: [{ date, count }],
        apiCalls: [{ date, hour, count }],
        featureUsage: [{ feature, count }]
    }
}
```

### 4.3 用户管理接口

```
GET /api/admin/users?page=1&limit=20&search=xxx
响应: {
    total: number,
    users: [{
        studentId, name, college, major, className,
        lastSync, pushEnabled, lastActive
    }]
}

GET /api/admin/users/:studentId
响应: {
    info: Student,
    stats: { courses, grades, exams },
    activities: [{ time, action }],
    pushRecords: PushLog[]
}
```

### 4.4 数据统计接口

```
GET /api/admin/data/overview
响应: {
    students: { count, size },
    courses: { count, size, bySemester },
    grades: { count, size, bySemester },
    exams: { count, size },
    plans: { count, size },
    pushTokens: { active, inactive }
}
```

### 4.5 系统监控接口

```
GET /api/admin/system/status
响应: {
    server: { status, uptime, port },
    database: { status, responseTime },
    pushService: { status, responseTime },
    jwxtConnection: { status, responseTime },
    mode: { isMock, mode }
}

GET /api/admin/system/logs?level=ERROR&start=xxx&end=xxx
响应: {
    logs: [{ id, level, module, message, time }]
}
```

### 4.6 API统计接口

```
GET /api/admin/api/stats
响应: {
    endpoints: [{
        endpoint, method, calls, avgTime, errorRate
    }]
}
```

### 4.7 推送管理接口

```
GET /api/admin/push/stats
响应: {
    todayCount: number,
    successRate: number,
    byType: [{ type, count }],
    recent: PushLog[]
}

GET /api/admin/push/logs?page=1&limit=50
响应: {
    total: number,
    logs: PushLog[]
}
```

---

## 5. 页面布局设计

### 5.1 整体布局
```
┌─────────────────────────────────────────────────────────┐
│  LOGO  开发者管理后台              [通知] [用户] [退出]  │
├──────────┬──────────────────────────────────────────────┤
│          │                                              │
│  仪表盘  │              主内容区域                       │
│          │                                              │
│  用户管理│                                              │
│          │                                              │
│  数据统计│                                              │
│          │                                              │
│  系统监控│                                              │
│          │                                              │
│  API管理 │                                              │
│          │                                              │
│  推送管理│                                              │
│          │                                              │
│  系统设置│                                              │
│          │                                              │
└──────────┴──────────────────────────────────────────────┘
```

### 5.2 响应式设计
- 桌面端：完整侧边栏 + 内容区
- 平板端：可折叠侧边栏 + 内容区
- 移动端：抽屉式侧边栏 + 内容区

---

## 6. 安全设计

### 6.1 认证与授权
- 使用JWT进行身份认证
- 区分超级管理员和普通管理员权限
- Token有效期：24小时，支持刷新

### 6.2 访问控制
- 所有管理接口需要管理员Token
- 敏感操作需要二次验证
- IP白名单可选配置

### 6.3 数据安全
- 用户密码不明文展示
- 敏感数据脱敏处理
- 操作日志完整记录

---

## 7. 开发计划

### Phase 1：基础框架（Week 1）
- [ ] 搭建Vue 3 + Element Plus项目
- [ ] 实现登录页面和JWT认证
- [ ] 完成基础布局组件

### Phase 2：核心功能（Week 2）
- [ ] 仪表盘页面开发
- [ ] 用户管理页面开发
- [ ] 数据统计页面开发

### Phase 3：高级功能（Week 3）
- [ ] 系统监控页面开发
- [ ] API管理页面开发
- [ ] 推送管理页面开发

### Phase 4：完善优化（Week 4）
- [ ] 系统设置页面开发
- [ ] 性能优化
- [ ] 安全加固
- [ ] 部署上线

---

## 8. 部署方案

### 8.1 部署架构
```
┌─────────────────┐     ┌─────────────────┐
│   管理后台前端   │────▶│   Nginx 反向代理  │
│   (静态资源)     │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │ 主API服务  │ │ 管理API   │ │  数据库   │
            │ :3000     │ │ :3000     │ │ MariaDB   │
            └───────────┘ └───────────┘ └───────────┘
```

### 8.2 部署方式
- 管理后台前端：构建后部署到Nginx静态目录
- 管理API：复用现有server.js，添加/admin路由
- 数据库：复用现有MariaDB实例

---

## 9. 附录

### 9.1 现有数据模型参考
```
Student: studentId, name, gender, enrollmentYear, className, major, college, lastSync
Course: id, studentId, semester, name, dayOfWeek, week, period, teacher, weeks, location
Grade: id, studentId, semester, courseCode, courseName, score, credit, gradePoint
Exam: id, studentId, courseName, examTime, location, seatNumber, examType, status
Plan: id, studentId, semester, courseCode, courseName, teachingUnit, credit
Progress: id, studentId, category, requiredCredits, completedCredits, remainingCredits
UserPushToken: id, studentId, pushToken, deviceInfo, isActive, lastActiveAt
ElectricityReminder: id, studentId, enabled, threshold, electricityAccount
```

### 9.2 现有API端点参考
```
POST /api/sync              - 同步学生数据
GET  /api/version           - 获取版本信息
GET  /api/mode/info         - 获取运行模式
POST /api/push/register     - 注册推送Token
POST /api/push/unregister   - 注销推送Token
GET  /api/emptyroom/...     - 空教室查询
GET  /api/electricity/...   - 电费查询
```
