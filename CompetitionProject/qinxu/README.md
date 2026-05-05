# 青序 (qinxu)

<div align="center">
  <h3>专为学生设计的 HarmonyOS 原生校园服务应用</h3>
  <p>HarmonyOS 手机应用 | 校园服务平台</p>
</div>

## 应用概述

**青序**是一款专为学生设计的 HarmonyOS 原生校园服务应用。通过整合教务系统数据，为学生提供一站式校园信息服务，帮助学生高效管理学习生活。

| 属性     | 信息                    |
| ------ | --------------------- |
| 应用名称   | 青序                   |
| 版本     | 1.0.0                 |
| 目标平台   | HarmonyOS (手机/平板)     |
| 开发语言   | ArkTS (TypeScript 超集) |
| 目标 SDK | 6.0.2(22)             |
| 包名     | com.example.qinxu     |

## 项目预览

### 功能模块

| 模块 | 功能描述 | 特色 |
|------|----------|------|
| 首页 | 智能问候、今日课程、明日课程、快捷功能入口 | 实时课程状态显示、课程冲突检测、渐变色主题头部设计 |
| 课表 | 周视图课表、周次切换、农历显示、课程详情 | 课程类型标识、课程冲突提示、本地缓存支持离线查看 |
| 成绩查询 | 学期分组、搜索功能、成绩详情、智能颜色标识 | 展开/收起动画效果、列表项渐入动画、搜索实时过滤 |
| 考试安排 | 考试列表、状态标识、详细信息 | 考试时间地点提醒、考试状态实时更新 |
| 培养计划 | 培养方案课程列表、课程分类展示、学分要求统计 | 课程规划可视化、培养方案详情查看 |
| 学分进度 | 学分分类统计、进度条可视化、已修/待修课程明细 | 学分完成情况一目了然、学分要求对比分析 |
| 个人中心 | 个人信息展示、账号设置、隐私设置、主题设置 | 明暗主题切换、安全退出登录、个人信息管理 |

---

## 核心功能

### 1. 首页

首页是用户进入应用后的主要信息展示区域，提供个性化的学习日程概览。

#### 功能特性

- **智能问候**：根据当前时段显示不同的诗意问候语（如"晨光熹微"、"日上三竿"、"夕阳西斜"等）
- **今日课程**：展示当天课程安排，支持实时状态显示
  - 进行中 - 当前正在上课
  - 即将上课 - 15分钟内开始
  - 即将下课 - 5分钟内结束
  - 已结束 - 课程已结束
- **明日课程**：提前预览第二天的课程安排
- **课程冲突检测**：自动检测并提示时间冲突的课程
- **快捷功能入口**：一键访问成绩查询、培养计划、学分进度、考试安排

#### 技术亮点

- 实时更新机制（每分钟刷新课程状态）
- 课程冲突智能检测算法
- 渐变色主题头部设计

---

### 2. 课表查询

功能完善的课程表管理系统，支持多周切换和详细信息查看。

#### 功能特性

- **周视图课表**：清晰的网格布局展示一周课程
- **周次切换**：支持快速切换查看任意教学周
- **农历显示**：集成农历日期和节假日信息
- **课程详情**：点击课程卡片查看完整信息
- **课程类型标识**：
  - 培养计划课程 - 正常修读的培养方案课程
  - 重修课程 - 重新修读的课程
  - 辅修课程 - 辅修专业课程
  - 其他课程 - 其他类型课程
- **课程冲突提示**：高亮显示时间冲突的课程
- **自定义设置**：
  - 开学日期设置
  - 总周数配置
  - 是否显示周末
  - 是否高亮当前周

#### 技术亮点

- 课程索引优化（O(1)时间复杂度查询）
- Swiper 组件实现流畅的周切换动画
- 本地缓存支持离线查看
- 历史版本数据管理

---

### 3. 成绩查询

全面的成绩信息查询与展示系统。

#### 功能特性

- **学期分组**：按学期自动归类成绩记录
- **搜索功能**：支持按课程名称快速搜索
- **成绩详情**：
  - 课程名称与编码
  - 成绩分数（支持数字和等级制）
  - 绩点信息
  - 学分信息
  - 课程类型
- **智能颜色标识**：
  - 优秀 (>=90分 / 优)
  - 良好 (80-89分 / 良)
  - 中等 (70-79分 / 中)
  - 及格 (60-69分 / 及格)
  - 不及格 (<60分 / 不合格)

#### 技术亮点

- 展开/收起动画效果
- 列表项渐入动画
- 搜索实时过滤

---

### 4. 考试安排

考试信息查询与提醒系统，帮助学生及时了解考试安排。

#### 功能特性

- **考试列表**：展示所有考试安排，按时间顺序排序
- **状态标识**：
  - 未开始
  - 进行中
  - 已结束
- **详细信息**：
  - 课程名称与编码
  - 考试时间（具体到分钟）
  - 考试地点（教学楼、教室）
  - 座位号
  - 考试场次
- **考试提醒**：重要考试前发送提醒通知
- **考试状态更新**：实时更新考试状态
- **考试信息筛选**：支持按学期、课程类型等筛选考试

---

### 5. 培养计划

学生培养方案与课程规划查看系统，帮助学生了解培养目标和课程要求。

#### 功能特性

- **培养方案课程列表**：完整展示专业培养方案中的所有课程
- **课程分类展示**：按课程类型（必修、选修、通识等）分类显示
- **学分要求统计**：详细展示各类课程的学分要求和完成情况
- **课程详情查看**：点击课程查看详细信息，包括课程描述、学分、学时等
- **培养目标展示**：清晰展示专业培养目标和毕业要求

---

### 6. 学分进度

学分完成情况可视化展示，帮助学生了解毕业进度。

#### 功能特性

- **学分分类统计**：按课程类型统计已修和未修学分
- **进度条可视化**：直观展示各类型学分的完成进度
- **已修/待修课程明细**：详细列出已修和待修的具体课程
- **毕业要求对比**：与毕业要求进行对比，明确差距
- **学分预警**：当学分完成情况不达标时提供预警提示

---

### 7. 个人中心

用户信息管理与系统设置。

#### 功能特性

- **个人信息展示**：
  - 姓名、性别
  - 学号
  - 入学年份
  - 学院、专业、班级
- **账号设置**：管理登录凭证
- **隐私设置**：隐私权限管理
- **主题设置**：明暗主题切换
- **推送设置**：管理应用推送通知
- **关于我们**：应用信息
- **退出登录**：安全退出账号

---

### 8. AI学习伙伴

智能学习辅助系统，提供个性化学习支持。

#### 功能特性

- **学习问题解答**：智能回答学习相关问题
- **学习计划制定**：根据课程安排生成学习计划
- **知识点总结**：自动总结课程知识点
- **学习进度跟踪**：监控学习进度和效果

#### 技术亮点

- 集成AI对话能力
- 个性化学习推荐算法
- 实时学习数据分析

---

### 9. 校园地图

校园地理信息系统，帮助学生熟悉校园环境。

#### 功能特性

- **校园地图浏览**：交互式校园地图
- **建筑物定位**：快速定位校园建筑
- **导航功能**：从当前位置导航到目标地点
- **场所分类**：按功能分类展示校园场所

#### 技术亮点

- 集成华为 Map Kit 地图服务
- 定位服务集成
- 校园设施图标分类（行政、餐饮、宿舍、图书馆、医疗、购物、体育、教学等）

---

### 10. 空教室查询

实时查询空闲教室信息，方便学生自习。

#### 功能特性

- **实时空教室列表**：显示当前空闲的教室
- **时间段筛选**：按时间段查询空教室
- **教学楼筛选**：按教学楼查询空教室
- **座位数量显示**：显示教室可容纳人数

#### 技术亮点

- 实时数据同步
- 高效查询算法
- 缓存优化

---

### 11. 课程提醒设置

课程提醒管理系统，确保学生不会错过课程。

#### 功能特性

- **提醒时间设置**：自定义课程开始前的提醒时间
- **提醒方式选择**：选择通知或弹窗提醒
- **提醒状态管理**：开启/关闭特定课程的提醒
- **系统限制提示**：显示系统提醒数量限制

#### 技术亮点

- 智能提醒调度
- 系统限制处理
- 用户友好的设置界面

---

### 12. 电费提醒设置

宿舍电费监控与提醒系统。

#### 功能特性

- **提醒开关**：开启/关闭电费余额提醒
- **余额阈值设置**：自定义低余额提醒阈值
- **宿舍信息绑定**：关联宿舍房间信息

---

### 13. 取水服务

校园取水设备扫码服务。

#### 功能特性

- **扫码取水**：扫描设备二维码取水
- **设备绑定**：绑定常用取水设备
- **冷热水开关**：选择出水温度
- **余额查询**：查看账户余额

---

## 登录系统

### 功能特性

- **账号密码登录**：使用学号和密码登录
- **记住密码**：安全保存登录凭证
- **自动登录**：支持开启自动登录功能
- **数据同步**：登录后自动同步教务数据

### 安全特性

- AES 加密存储密码
- 本地凭证安全保存
- 支持手动清除登录状态

---

## 用户界面

### 设计风格

- **沉静光感设计**：对齐鸿蒙6.1沉静光感设计语言
- **沉静毛玻璃**：低模糊度 + 材质感半透明背景
- **微光高光线**：模拟自然光照射的细腻顶部光线
- **弝散阴影**：大半径低透明度柔和投影
- **光感指示器**：选中态底部光点指示
- **圆角卡片**：统一使用圆角卡片布局
- **渐变主题**：头部区域使用渐变色设计

### 主题支持

- 浅色模式
- 深色模式
- 跟随系统

### 交互动画

- 沉静弹簧曲线动效（更柔和的过渡）
- 图标缩放过渡（22↔24vp）
- 按压反馈（0.94缩放回弹）
- 页面切换动画
- 列表项渐入效果
- 展开/收起动画

---

## 项目结构

```
qinxu/
├── AppScope/                              # 应用全局资源
│   ├── app.json5                          # 应用全局配置
│   └── resources/
│       └── base/element/string.json       # 应用级字符串资源
│
├── entry/                                 # 主模块
│   ├── src/main/
│   │   ├── ets/                           # ArkTS 源码
│   │   │   │
│   │   │   ├── common/constants/          # 常量定义
│   │   │   │   ├── ApiConstants.ets       # API 相关常量
│   │   │   │   ├── AppConstants.ets       # 应用常量
│   │   │   │   ├── CourseConstants.ets    # 课程相关常量
│   │   │   │   └── index.ets              # 常量导出
│   │   │   │
│   │   │   ├── components/                # 可复用组件
│   │   │   │   ├── AccessibilitySupport.ets    # 无障碍支持组件
│   │   │   │   ├── CourseCard.ets              # 课程卡片组件
│   │   │   │   ├── DormitoryCard.ets           # 宿舍卡片组件
│   │   │   │   ├── GlassMenubar.ets            # 沉静光感底部导航栏组件
│   │   │   │   ├── PerformanceOptimizedList.ets # 性能优化列表组件
│   │   │   │   ├── PrivacyDialog.ets           # 隐私对话框组件
│   │   │   │   ├── ScheduleTabs.ets            # 课表标签页组件
│   │   │   │   ├── SemesterSettings.ets        # 学期设置组件
│   │   │   │   ├── StudyAssistant.ets          # 学习助手组件
│   │   │   │   ├── TimeSlotEditor.ets          # 时间段编辑器组件
│   │   │   │   └── TrainingPlanAssistant.ets   # 培养计划助手组件
│   │   │   │
│   │   │   ├── entryability/              # 应用入口
│   │   │   │   ├── EntryAbility.ets       # 应用入口能力
│   │   │   │   └── EntryBackupAbility.ets # 备份恢复能力
│   │   │   │
│   │   │   ├── hooks/                     # 自定义钩子
│   │   │   │   ├── useDebounce.ets        # 防抖钩子
│   │   │   │   ├── useLoading.ets         # 加载状态钩子
│   │   │   │   ├── useThrottle.ets        # 节流钩子
│   │   │   │   └── index.ets              # 钩子导出
│   │   │   │
│   │   │   ├── model/                     # 数据模型
│   │   │   │   └── SettingsData.ets       # 设置数据模型
│   │   │   │
│   │   │   ├── pages/                     # 页面组件
│   │   │   │   ├── Index.ets              # 登录入口页面
│   │   │   │   ├── MainPage.ets           # 主页面框架(底部导航)
│   │   │   │   ├── HomePage.ets           # 首页
│   │   │   │   ├── SchedulePage.ets       # 课程表页面
│   │   │   │   ├── ScheduleSettingsPage.ets      # 课表设置页面
│   │   │   │   ├── GradeQueryPage.ets     # 成绩查询页面
│   │   │   │   ├── CreditProgressPage.ets # 学分进度页面
│   │   │   │   ├── ExamPage.ets           # 考试安排页面
│   │   │   │   ├── TrainingPlanPage.ets   # 培养计划页面
│   │   │   │   ├── ProfilePage.ets        # 个人中心页面
│   │   │   │   ├── AccountSettingsPage.ets      # 账号设置页面
│   │   │   │   ├── ThemeSettingsPage.ets  # 主题设置页面
│   │   │   │   ├── PrivacySettingsPage.ets      # 隐私设置页面
│   │   │   │   ├── PrivacyPolicyPage.ets  # 隐私政策页面
│   │   │   │   ├── AboutPage.ets          # 关于页面
│   │   │   │   ├── CommonPage.ets         # 通用设置页面
│   │   │   │   ├── AIStudyPartnerPage.ets # AI学习伙伴页面
│   │   │   │   ├── CampusMapPage.ets      # 校园地图页面
│   │   │   │   ├── CourseReminderSettingsPage.ets # 课程提醒设置页面
│   │   │   │   ├── ElectricityReminderSettingsPage.ets # 电费提醒设置页面
│   │   │   │   ├── EmptyRoomPage.ets      # 空教室查询页面
│   │   │   │   ├── FetchWater.ets         # 取水页面
│   │   │   │   └── PushSettingsPage.ets   # 推送设置页面
│   │   │   │
│   │   │   ├── services/                  # 业务服务
│   │   │   │   ├── CourseReminderService.ets   # 课程提醒服务
│   │   │   │   ├── PushNotificationService.ets # 推送通知服务
│   │   │   │   ├── StorageService.ets     # 存储服务
│   │   │   │   ├── StudentService.ets     # 学生数据服务
│   │   │   │   └── index.ets              # 服务导出
│   │   │   │
│   │   │   ├── types/                     # 类型定义
│   │   │   │   ├── academic.ets           # 学术相关类型
│   │   │   │   ├── common.ets             # 通用类型定义
│   │   │   │   └── index.ets              # 类型导出
│   │   │   │
│   │   │   └── utils/                     # 工具类
│   │   │       ├── AIService.ets          # AI 服务工具
│   │   │       ├── ApiClient.ets          # API 请求封装
│   │   │       ├── CryptoUtils.ets        # AES 加密工具
│   │   │       ├── DataStore.ets          # 数据存储工具
│   │   │       ├── LocalDataManager.ets   # 本地数据管理器
│   │   │       ├── LunarCalendar.ets      # 农历日历工具
│   │   │       ├── NotificationManager.ets     # 通知管理器
│   │   │       ├── ScheduleDataSource.ets # 课程表数据源
│   │   │       ├── SemesterUtils.ets      # 学期工具类
│   │   │       ├── ThemeColorManager.ets  # 主题颜色管理器
│   │   │       ├── WeatherService.ets     # 天气服务工具
│   │   │       └── XyyxtApiClient.ets     # 校园一体化 API 客户端
│   │   │
│   │   └── resources/                     # 资源文件
│   │       ├── base/
│   │       │   ├── element/
│   │       │   │   ├── color.json         # 颜色资源定义
│   │       │   │   ├── float.json         # 浮点数资源
│   │       │   │   └── string.json        # 字符串资源
│   │       │   ├── media/                 # 图标资源(68个)
│   │       │   │   ├── R_C.PNG            # 应用图标
│   │       │   │   ├── house.svg          # 首页图标(未选中)
│   │       │   │   ├── house_fill.svg     # 首页图标(选中)
│   │       │   │   ├── calendar.svg       # 日历图标(未选中)
│   │       │   │   ├── calendar_fill.svg  # 日历图标(选中)
│   │       │   │   ├── person_2.svg       # 个人图标(未选中)
│   │       │   │   ├── person_2_fill.svg  # 个人图标(选中)
│   │       │   │   ├── ic_*.svg           # 校园设施分类图标
│   │       │   │   └── ...                # 其他功能图标
│   │       │   └── profile/
│   │       │       ├── main_pages.json    # 页面路由配置
│   │       │       ├── backup_config.json # 备份配置
│   │       │       └── network_config.json # 网络配置
│   │       └── dark/
│   │           └── element/color.json     # 暗色主题颜色
│   │
│   ├── src/test/                          # 单元测试
│   │   ├── List.test.ets                  # 列表测试
│   │   └── LocalUnit.test.ets             # 本地单元测试
│   │
│   ├── src/ohosTest/ets/test/             # UI测试
│   │   ├── Ability.test.ets               # 能力测试
│   │   └── List.test.ets                  # 列表测试
│   │
│   ├── build-profile.json5                # 模块构建配置
│   ├── module.json5                       # 模块配置
│   ├── oh-package.json5                   # 模块依赖
│   ├── hvigorfile.ts                      # 模块构建脚本
│   └── obfuscation-rules.txt              # 代码混淆规则
│
├── build-profile.json5                    # 全局构建配置
├── oh-package.json5                       # 项目依赖配置
├── oh-package-lock.json5                  # 依赖锁定文件
├── code-linter.json5                      # 代码检查配置
├── hvigorfile.ts                          # Hvigor 构建脚本
├── local.properties                       # 本地属性配置
├── .gitignore                             # Git 忽略配置
└── README.md                              # 项目说明文档
```

### 文件统计

| 类别    | 数量   | 说明            |
| ----- | ---- | ------------- |
| 页面文件  | 24 个 | 应用各功能页面       |
| 组件文件  | 11 个 | 可复用 UI 组件     |
| 工具类   | 12 个 | 通用工具函数        |
| 服务类   | 5 个  | 业务逻辑封装        |
| 类型定义  | 3 个  | TypeScript 类型 |
| 自定义钩子 | 4 个  | 状态管理钩子        |
| 常量定义  | 4 个  | 应用常量配置        |
| 图标资源  | 68 个 | SVG/PNG 图标    |

---

## 分页说明

### 底部导航栏

应用采用**沉静光感底部导航栏**设计，对齐鸿蒙6.1设计语言，包含三个主要页面：

| Tab 图标 | 页面名称         | 功能描述             |
| ------ | ------------ | ---------------- |
| 首页     | HomePage     | 今日课程、明日课程、快捷功能入口 |
| 课表     | SchedulePage | 周视图课表、周次切换、课程详情  |
| 我的     | ProfilePage  | 个人信息、系统设置、账号管理   |

导航栏特性：
- 沉静毛玻璃背景（backdropBlur 28）
- 微光高光线（60%宽度居中，模拟自然光照射）
- 弝散双层阴影（大半径低透明度柔和投影）
- 光感指示器（选中态底部光点）
- 沉静弹簧曲线动效（stiffness:280, damping:38）
- 按压反馈（0.94缩放回弹）
- 亮色/暗色模式自适应

### 页面层级结构

```
Level 1: MainPage (底部导航容器)
    │
    ├── Level 2: HomePage (首页)
    │       ├── Level 3: GradeQueryPage (成绩查询)
    │       ├── Level 3: TrainingPlanPage (培养计划)
    │       ├── Level 3: CreditProgressPage (学分进度)
    │       ├── Level 3: ExamPage (考试安排)
    │       ├── Level 3: AIStudyPartnerPage (AI学习伙伴)
    │       ├── Level 3: CampusMapPage (校园地图)
    │       └── Level 3: EmptyRoomPage (空教室查询)
    │
    ├── Level 2: SchedulePage (课表)
    │       ├── Level 3: ScheduleSettingsPage (课表设置)
    │       └── Level 3: CourseReminderSettingsPage (课程提醒设置)
    │
    └── Level 2: ProfilePage (我的)
            ├── Level 3: AccountSettingsPage (账号设置)
            ├── Level 3: PrivacySettingsPage (隐私设置)
            ├── Level 3: ThemeSettingsPage (主题设置)
            ├── Level 3: AboutPage (关于我们)
            ├── Level 3: PrivacyPolicyPage (隐私政策)
            ├── Level 3: PushSettingsPage (推送设置)
            ├── Level 3: ElectricityReminderSettingsPage (电费提醒设置)
            └── Level 3: CommonPage (通用设置)
```

### 导航交互

- **底部 Tab 切换**：点击底部图标切换主页面，支持滑动切换
- **页面跳转**：使用 `router.pushUrl()` 进行页面跳转
- **返回操作**：使用 `router.back()` 返回上一页
- **首页快捷入口**：点击卡片直接跳转对应功能页面

---

## 技术架构

### 技术栈

| 类别   | 技术                                 |
| ---- | ---------------------------------- |
| 开发语言 | ArkTS (TypeScript 超集)              |
| UI框架 | ArkUI 声明式开发范式                      |
| 状态管理 | @State, @Prop, @Link, @StorageLink |
| 网络请求 | @ohos.net.http                     |
| 数据存储 | @kit.ArkData (Preferences)         |
| 加密安全 | AES 加密 (CryptoUtils)              |
| 构建工具 | Hvigor                             |
| 测试框架 | @ohos/hypium                       |

### 华为服务集成

| 服务 | 用途 |
|------|------|
| Push Kit | 推送通知服务 |
| Map Kit | 校园地图服务 |
| Device Status Detection | 设备状态检测 |
| Safety Detect | 安全检测 |

### 核心模块

#### 1. API 客户端

```typescript
// 主要接口
- syncStudent()       // 登录并同步数据
- getStudentData()    // 获取学生数据
- getLatestSemester() // 获取最新学期
```

#### 2. 数据存储

```typescript
// 主要功能
- 本地设置存储
- 学生信息缓存
- 课程数据管理
- 培养计划存储
```

#### 3. 主题管理

```typescript
// 主要功能
- 主题色配置
- 明暗模式切换
- 渐变色管理
```

#### 4. 通知管理

```typescript
// 主要功能
- 课程提醒
- 考试安排提醒
- 电费余额提醒
- 推送通知
```

#### 5. 加密工具

```typescript
// 主要功能
- AES 加密/解密
- 密钥管理
- 安全数据传输
```

#### 6. 自定义钩子

```typescript
// useDebounce - 防抖钩子
// 作用: 延迟执行函数，连续触发时只执行最后一次
// 应用场景: 搜索框输入、表单验证

// useThrottle - 节流钩子
// 作用: 限制函数执行频率，指定间隔内只执行一次
// 应用场景: 按钮点击防重复、滚动事件优化

// useLoading - 加载状态钩子
// 作用: 管理加载状态，统一控制加载提示
// 应用场景: 数据请求、表单提交
```

---

## 权限说明

| 权限                                | 用途              |
| --------------------------------- | --------------- |
| ohos.permission.INTERNET          | 网络请求，获取教务数据     |
| ohos.permission.ACCELEROMETER     | 加速度传感器          |
| ohos.permission.APPROXIMATELY_LOCATION | 大致位置信息       |
| ohos.permission.LOCATION          | 精确位置信息（校园地图导航）  |
| ohos.permission.PUBLISH_AGENT_REMINDER | 后台代理提醒（课程/考试提醒） |
| ohos.permission.CAMERA            | 相机（扫码取水）        |
| ohos.permission.DETECT_GESTURE    | 手势检测            |

---

## 页面导航

```
Index (登录页)
    │
    ▼
MainPage (主页 - 沉静光感底部导航)
    │
    ├── HomePage (首页)
    │       ├── GradeQueryPage (成绩查询)
    │       ├── TrainingPlanPage (培养计划)
    │       ├── CreditProgressPage (学分进度)
    │       ├── ExamPage (考试安排)
    │       ├── AIStudyPartnerPage (AI学习伙伴)
    │       ├── CampusMapPage (校园地图)
    │       └── EmptyRoomPage (空教室查询)
    │
    ├── SchedulePage (课表)
    │       ├── ScheduleSettingsPage (课表设置)
    │       └── CourseReminderSettingsPage (课程提醒设置)
    │
    └── ProfilePage (我的)
            ├── AccountSettingsPage (账号设置)
            ├── PrivacySettingsPage (隐私设置)
            ├── ThemeSettingsPage (主题设置)
            ├── AboutPage (关于我们)
            ├── PrivacyPolicyPage (隐私政策)
            ├── PushSettingsPage (推送设置)
            ├── ElectricityReminderSettingsPage (电费提醒设置)
            └── CommonPage (通用设置)
```

---

## 构建与运行

### 环境要求

- DevEco Studio 4.0+
- HarmonyOS SDK 6.0.2(22)
- Node.js 14+

### 构建命令

```bash
# 构建项目
hvigorw assembleHap

# 清理构建
hvigorw clean
```

### 运行调试

1. 在 DevEco Studio 中打开项目
2. 连接 HarmonyOS 设备或启动模拟器
3. 点击运行按钮

---

## 开发规范

### 命名规范

- **页面文件**: PascalCase + `Page.ets` (如 `HomePage.ets`)
- **组件文件**: PascalCase + `.ets` (如 `CourseCard.ets`)
- **变量/函数**: camelCase (如 `getCurrentWeek`)
- **常量**: UPPER\_SNAKE\_CASE (如 `BASE_URL`)

### 代码规范

- 使用 TypeScript 严格模式
- 遵循 ArkTS 编码规范
- 组件代码不超过 300 行
- 使用 async/await 处理异步操作

---

## 更新日志

### v1.0.0

- 首次发布
- 实现登录认证功能
- 实现课表查询功能
- 实现成绩查询功能
- 实现考试安排功能
- 实现培养计划功能
- 实现学分进度功能
- 实现主题切换功能
- 实现个人中心功能
- 实现沉静光感底部导航栏（对齐鸿蒙6.1设计语言）
- 实现电费提醒设置
- 实现取水服务
- 集成华为 Push Kit / Map Kit / Safety Detect

---

## 安装说明

### 方法一：通过 DevEco Studio 安装

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/your-username/qinxu.git
   ```

2. 在 DevEco Studio 中打开项目

3. 连接 HarmonyOS 设备或启动模拟器

4. 点击运行按钮进行安装

### 方法二：直接安装 HAP 包

1. 从 [Releases](https://github.com/your-username/qinxu/releases) 页面下载最新的 HAP 包

2. 将 HAP 包传输到 HarmonyOS 设备

3. 在设备上点击 HAP 包进行安装

---

## 贡献指南

我们欢迎社区贡献！如果您想参与项目开发，请按照以下步骤操作：

1. Fork 本仓库

2. 创建特性分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. 提交您的更改：
   ```bash
   git commit -m 'Add some feature'
   ```

4. 推送到分支：
   ```bash
   git push origin feature/your-feature-name
   ```

5. 打开 Pull Request

### 开发规范

- 遵循项目的命名规范和代码风格
- 确保代码通过类型检查
- 为新功能添加适当的测试
- 提交清晰、有意义的 commit 信息

---

## 技术支持

如果您在使用过程中遇到问题，请通过以下方式获取支持：

- **邮件联系**：yelan12192023@163.com

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 致谢

感谢所有为项目做出贡献的开发者和用户！

---

<div align="center">
  <p>&copy; 2026 青序团队</p>
</div>
