# JW-Backend 项目文件结构

## 根目录
- `.gitignore`
- `README.md`
- `package-lock.json`
- `package.json`
- `server.js`
- `test_water.js`

## 文档目录
### docs/
- `PROJECT_PLAN.md`

## 源代码目录
### src/
- `index.js`

#### API 模块
##### src/api/
- `announcement.js`
- `auth.js`
- `electricity.js`
- `emptyroom.js`
- `student.js`
- `water.js`

#### 数据库模块
##### src/db/
- `index.js`
- `sync.js`

###### 数据库模型
####### src/db/models/
- `index.js`

#### 解析器模块
##### src/parser/
- `index.js`

#### 服务模块
##### src/services/
- `electricityMonitor.js`
- `notificationMonitor.js`
- `pushService.js`
- `realtimePush.js`

#### 工具模块
##### src/utils/
- `constants.js`
- `encryption.js`
- `request.js`

#### 校园系统模块
##### src/xyyxt/
- `auth.js`
- `constants.js`
- `guilinElec.js`
- `index.js`

## 测试目录
### tests/
- `cli.js`
- `test_announcement.js`
- `test_api.py`
- `test_attachment.js`
- `test_db.js`
- `test_detail.js`
- `test_encryption.js`
- `test_new.js`
- `test_parse_timetable.js`

#### 测试资源
##### tests/fixtures/
- `timetable_sample.html`
