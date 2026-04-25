# 模拟数据模式使用说明

## 📋 概述

本项目已成功改造为支持**模拟数据模式**，无需连接真实的教务系统即可进行开发和测试。

## 🚀 快速启动

### 1. 使用模拟数据模式（推荐用于开发/测试）

```bash
# 确保环境变量设置正确
cat .env
# 应该包含：
# MOCK_MODE=true
# NODE_ENV=development

# 启动服务
npm start
```

### 2. 切换到生产模式（使用真实数据）

```bash
# 修改 .env 文件
MOCK_MODE=false

# 重启服务
npm start
```

## 📊 可用的测试账号

| 学号 | 密码 | 姓名 | 班级 | 专业 |
|------|------|------|------|------|
| 202101001 | 123456 | 张三 | 计算机2101班 | 计算机科学与技术 |
| 202101002 | 123456 | 李四 | 计算机2101班 | 计算机科学与技术 |
| 202102001 | 123456 | 王五 | 软件2101班 | 软件工程 |
| 202103001 | 123456 | 赵六 | 人工智能2101班 | 人工智能 |
| 202201001 | 123456 | 钱七 | 计算机2201班 | 计算机科学与技术 |

**默认密码：`123456`**

## 🔌 API 接口说明

### 1. 模式检测接口

```http
GET http://localhost:3000/api/mode/info
```

**响应示例（模拟模式）：**
```json
{
    "success": true,
    "mode": "mock-data",
    "isMock": true,
    "message": "当前使用模拟数据模式",
    "availableTestAccounts": [...],
    "endpoints": {...}
}
```

### 2. 教务系统登录与数据同步

```http
POST http://localhost:3000/api/sync
Content-Type: application/json

{
    "username": "202101001",
    "password": "123456"
}
```

**响应数据包括：**
- ✅ 学生基本信息（姓名、性别、班级、专业等）
- ✅ 课程表（8门课程）
- ✅ 成绩记录（2个学期）
- ✅ 考试安排（4门考试）
- ✅ 培养计划（2个学期）
- ✅ 学分进度（5个类别）

### 3. 校园一信通登录

```http
POST http://localhost:3000/api/xyyxt/login
Content-Type: application/json

{
    "username": "202101001",
    "password": "123456"
}
```

### 4. 宿舍信息查询

#### 获取宿舍楼列表
```http
GET http://localhost:3000/api/xyyxt/buildings?username=202101001&areaId=nnxq
```

**支持的校区：**
- `nnxq` - 南宁校区（13栋楼）
- `glxq` - 桂林校区（9栋楼）

#### 获取房间列表
```http
GET http://localhost:3000/api/xyyxt/rooms?username=202101001&buildingId=4320&page=1&size=100
```

#### 查询电费余额
```http
GET http://localhost:3000/api/xyyxt/electricity?username=202101001&roomId=H4320101
```

### 5. 其他接口

#### 用户信息
```http
GET http://localhost:3000/api/xyyxt/userinfo?username=202101001
```

#### 余额查询
```http
GET http://localhost:3000/api/xyyxt/balance?username=202101001
```

#### 交易记录
```http
GET http://localhost:3000/api/xyyxt/transactions?username=202101001&page=1&size=20
```

#### 消费记录
```http
GET http://localhost:3000/api/xyyxt/consumption?username=202101001&page=1&size=20
```

#### 充值记录
```http
GET http://localhost:3000/api/xyyxt/recharge?username=202101001&page=1&size=20
```

## 🏗️ 项目结构

```
src/
├── mockData.js          # 模拟数据定义
├── mode.js              # 模式检测模块
├── adapter.js           # 模式适配器（自动切换真实/模拟数据）
├── api/
│   ├── auth.js          # 教务系统认证（已改造为模拟）
│   └── student.js       # 学生数据接口（已改造为模拟）
└── xyyxt/
    ├── auth.js          # 校园一信通（已改造为模拟）
    └── guilinElec.js    # 桂林校区电费（已改造为模拟）

.env                     # 环境变量配置
.env.example             # 环境变量示例
server.js                # 主服务器文件（已更新）
```

## 🔧 配置说明

### 环境变量 (.env)

```env
# 服务器端口
PORT=3000

# ⭐ 核心配置：数据模式
# MOCK_MODE=true   - 使用模拟数据（开发/测试）
# MOCK_MODE=false  - 使用真实数据（生产环境）
MOCK_MODE=true

# 运行环境
NODE_ENV=development

# 数据库配置（仅在 MOCK_MODE=false 时需要）
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=jw_database
```

## 📝 前端适配指南

### 1. 检测当前模式

前端可以在应用启动时调用：

```javascript
async function checkMode() {
    const response = await fetch('/api/mode/info');
    const data = await response.json();
    
    if (data.isMock) {
        console.log('当前使用模拟数据模式');
        console.log('可用测试账号:', data.availableTestAccounts);
        // 可以显示提示信息给开发者
    }
    
    return data;
}
```

### 2. 根据模式调整UI

```javascript
if (mode.isMock) {
    // 显示测试账号选择器
    // 显示"模拟数据"标识
    // 禁用某些需要真实数据的功能
}
```

### 3. URL区分策略

**方案A：不同域名/端口**
- 开发环境：`http://localhost:3000` (MOCK_MODE=true)
- 生产环境：`https://api.yourschool.edu` (MOCK_MODE=false)

**方案B：同一服务器，通过API检测**
- 前端调用 `/api/mode/info` 动态判断
- 根据返回值调整行为

## 🧪 测试用例

### 测试登录成功
```bash
curl -X POST http://localhost:3000/api/sync \
  -H "Content-Type: application/json" \
  -d '{"username":"202101001","password":"123456"}'
```

### 测试密码错误
```bash
curl -X POST http://localhost:3000/api/sync \
  -H "Content-Type: application/json" \
  -d '{"username":"202101001","password":"wrong"}'
```
**预期结果：** `{"success": false, "message": "密码错误，请重新输入"}`

### 测试用户不存在
```bash
curl -X POST http://localhost:3000/api/sync \
  -H "Content-Type: application/json" \
  -d '{"username":"999999999","password":"123456"}'
```
**预期结果：** `{"success": false, "message": "该学号不存在"}`

## 📦 模拟数据详情

### 学生数据
- **5名虚拟学生**，涵盖不同年级和专业
- 完整的个人信息、班级、学院信息

### 课程表数据
- **8门课程**，覆盖周一至周五
- 包含必修课和选修课
- 详细的时间、地点、教师信息

### 成绩数据
- **2个学期**的成绩记录
- **11门课程**成绩
- 包含学分、绩点、考试类型

### 考试安排
- **4门考试**安排
- 详细的时间、地点、座位号

### 宿舍数据
- **南宁校区**：13栋宿舍楼
- **桂林校区**：9栋宿舍楼
- 每栋楼约120个房间（6层×20间/层）
- 电费余额随机生成

### 一信通数据
- 账户余额
- 交易记录
- 消费记录
- 充值记录

## ⚠️ 注意事项

1. **数据隔离**：模拟数据不会影响真实数据库
2. **性能优化**：模拟数据添加了适当的延迟（300-800ms）以模拟真实网络请求
3. **数据一致性**：每次请求返回的数据格式与真实API完全一致
4. **扩展性**：可在 [mockData.js](src/mockData.js) 中轻松添加更多测试数据

## 🔄 切换回真实数据

当需要使用真实数据时：

1. 修改 `.env` 文件：
   ```
   MOCK_MODE=false
   ```

2. 重启服务：
   ```bash
   npm start
   ```

3. 服务将自动切换到真实数据模式，从教务系统获取数据

## 💡 开发建议

- **开发阶段**：始终使用 `MOCK_MODE=true`，避免依赖外部系统
- **联调阶段**：前后端都使用模拟数据，确保接口对接无误
- **测试阶段**：使用固定的模拟数据编写自动化测试
- **部署前**：切换到 `MOCK_MODE=false` 进行真实环境验证

## 🐛 故障排除

### 问题：启动时数据库连接失败
**解决**：这是正常的，在模拟模式下不需要数据库连接。服务会以无数据库模式运行。

### 问题：API返回空数据
**检查**：
1. 确认 `.env` 中 `MOCK_MODE=true`
2. 重启服务使配置生效
3. 查看 `/api/mode/info` 确认模式状态

### 问题：端口被占用
**解决**：修改 `.env` 中的 `PORT` 配置，或停止占用该端口的进程

---

## 📞 技术支持

如有问题，请检查：
1. 服务器日志输出（控制台会显示当前运行模式）
2. `/api/mode/info` 接口返回的状态
3. `.env` 文件配置是否正确

---

**版本**: 1.0.0  
**最后更新**: 2026-04-22  
**状态**: ✅ 已完成并测试通过
