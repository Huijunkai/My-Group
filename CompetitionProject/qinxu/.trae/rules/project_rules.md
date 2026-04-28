# NNLG HarmonyOS 项目开发规范

## 项目概述
NNLG 是一款 HarmonyOS 原生应用，使用 ArkTS 语言开发，目标 SDK 版本为 6.0.2(22)。

## 技术栈
- **开发语言**: ArkTS (TypeScript 超集)
- **UI框架**: ArkUI 声明式开发范式
- **目标平台**: HarmonyOS 手机应用
- **构建工具**: Hvigor

## 项目结构

```
entry/src/main/
├── ets/
│   ├── common/                    # 公共资源
│   │   ├── constants/             # 常量定义
│   │   └── utils/                 # 公共工具函数
│   ├── components/                # 可复用UI组件
│   │   ├── AccessibilitySupport.ets
│   │   ├── CourseCard.ets
│   │   ├── GlassMenubar.ets
│   │   └── ...
│   ├── entryability/              # 应用入口能力
│   │   └── EntryAbility.ets
│   ├── hooks/                     # 自定义状态钩子
│   ├── model/                     # 数据模型定义
│   │   └── SettingsData.ets
│   ├── pages/                     # 页面组件
│   │   ├── Index.ets              # 入口页面
│   │   ├── MainPage.ets           # 主页面
│   │   └── ...
│   ├── services/                  # 业务服务层
│   ├── types/                     # TypeScript 类型定义
│   └── utils/                     # 工具类
│       ├── ApiClient.ets          # API 请求封装
│       ├── DataStore.ets          # 数据存储
│       └── ...
├── resources/                     # 资源文件
│   ├── base/
│   │   ├── element/               # 颜色、字符串等
│   │   ├── media/                 # 图片、图标
│   │   └── profile/               # 配置文件
│   └── dark/                      # 暗色主题资源
└── module.json5                   # 模块配置
```

## 命名规范

### 文件命名
- **页面文件**: PascalCase + `Page.ets` 后缀，如 `HomePage.ets`
- **组件文件**: PascalCase + `.ets` 后缀，如 `CourseCard.ets`
- **工具类**: PascalCase + `.ets` 后缀，如 `ApiClient.ets`
- **模型类**: PascalCase + `.ets` 后缀，如 `SettingsData.ets`
- **类型定义**: camelCase + `.ets` 后缀，如 `index.ets` (导出文件)

### 代码命名
- **组件名称**: PascalCase，如 `@Component struct CourseCard`
- **变量/函数**: camelCase，如 `getCurrentWeek()`
- **常量**: UPPER_SNAKE_CASE，如 `BASE_URL`
- **接口/类型**: PascalCase，接口以 `I` 开头可选，如 `StudentData`
- **私有成员**: 下划线前缀，如 `_privateMethod()`

## ArkTS 编码规范

### 组件定义
```typescript
@Component
export struct ComponentName {
  @State private stateVar: Type = initialValue;
  @Prop propVar: Type;
  @Link linkedVar: Type;
  
  build() {
    Column() {
      // UI 构建
    }
  }
}
```

### 页面定义
```typescript
@Entry
@Component
struct PageName {
  build() {
    Column() {
      // 页面内容
    }
    .width('100%')
    .height('100%')
  }
}
```

### 状态管理
- `@State`: 组件内部可变状态
- `@Prop`: 父组件传入的只读属性
- `@Link`: 双向绑定
- `@Observed` + `@ObjectLink`: 观察嵌套对象变化
- `@Watch`: 监听状态变化

### 样式规范
```typescript
Text('示例文本')
  .fontSize(16)
  .fontWeight(FontWeight.Medium)
  .fontColor('#1c1c1e')
  .maxLines(2)
  .textOverflow({ overflow: TextOverflow.Ellipsis })
```

## 资源引用规范

### 字符串资源
```typescript
Text($r('app.string.key_name'))
```

### 颜色资源
```typescript
.backgroundColor($r('app.color.primary'))
```

### 图片资源
```typescript
Image($r('app.media.icon_name'))
```

## API 请求规范

### 使用 ApiClient
```typescript
const response = await ApiClient.syncStudent(username, password);
if (response.success) {
  // 处理成功响应
} else {
  // 处理错误
}
```

### 错误处理
```typescript
try {
  const result = await ApiClient.getStudentData(studentId);
  // 处理结果
} catch (error) {
  console.error('请求失败:', JSON.stringify(error));
}
```

## 数据存储规范

### 使用 DataStore
```typescript
const dataStore = DataStore.getInstance();
await dataStore.initialize(context);
const value = await dataStore.get<string>('key', 'defaultValue');
await dataStore.set('key', 'value');
```

### 使用 LocalDataManager
```typescript
const localDataManager = LocalDataManager.getInstance();
await localDataManager.initialize(context);
```

## 主题适配

### 暗色模式支持
- 在 `resources/dark/element/color.json` 中定义暗色主题颜色
- 使用 `$r('app.color.xxx')` 引用颜色，系统自动切换

### 主题管理
```typescript
const themeColorManager = ThemeColorManager.getInstance();
await themeColorManager.initialize(context);
```

## 构建命令

### 构建项目
```bash
hvigorw assembleHap
```

### 清理构建
```bash
hvigorw clean
```

### 类型检查
项目使用 TypeScript 严格模式，确保类型安全。

## 代码检查

项目配置了以下代码检查规则:
- `plugin:@performance/recommended` - 性能优化规则
- `plugin:@typescript-eslint/recommended` - TypeScript 最佳实践
- 安全相关规则 (`@security/*`)

## 注意事项

1. **禁止在代码中硬编码敏感信息**（密钥、密码等）
2. **使用资源文件管理字符串**，便于国际化
3. **组件拆分原则**: 单个组件代码不超过 300 行
4. **异步操作**: 使用 `async/await` 而非回调
5. **内存管理**: 及时释放不再使用的资源
6. **性能优化**: 使用 LazyForEach 处理长列表

## 测试规范

### 单元测试
- 测试文件位置: `entry/src/test/`
- 测试框架: @ohos/hypium
- 命名规范: `*.test.ets`

### UI测试
- 测试文件位置: `entry/src/ohosTest/ets/test/`
