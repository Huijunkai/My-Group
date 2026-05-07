/**
 * 主题设置页面模块
 * 
 * 提供主题外观设置功能，包括：
 * - 主题模式切换（浅色/深色/跟随系统）
 * - 主题颜色自定义
 * - 动效手势开关
 * 
 * @module ThemeSettingsPage
 */

import { router, promptAction } from '@kit.ArkUI';
import { preferences } from '@kit.ArkData';
import { common, ConfigurationConstant } from '@kit.AbilityKit';
import { ThemeColorManager, ThemeColors, PRESET_COLORS } from '../utils/ThemeColorManager';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'theme_preferences';
/** 主题模式存储键名 */
const KEY_THEME_MODE = 'themeMode';
/** 动效手势存储键名 */
const KEY_MOTION_GESTURE = 'motionGesture';

/** 主题模式类型 */
type ThemeMode = 'light' | 'dark' | 'system';

/**
 * 主题选项接口
 */
interface ThemeOption {
  /** 模式 */
  mode: ThemeMode;
  /** 标题 */
  title: string;
  /** 描述 */
  description: string;
  /** 图标 */
  icon: string;
}

/**
 * 颜色目标接口
 */
interface ColorTarget {
  /** 键名 */
  key: string;
  /** 标题 */
  title: string;
  /** 描述 */
  description: string;
}

/**
 * 主题设置页面组件
 * 
 * 配置应用的主题外观
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/ThemeSettingsPage' });
 * ```
 */
@Entry
@Component
struct ThemeSettingsPage {
  /** 当前主题模式 */
  @State currentTheme: ThemeMode = 'system';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 头部渐变起始色 */
  @State headerGradientStart: string = '#4A90D9';
  /** 头部渐变结束色 */
  @State headerGradientEnd: string = '#0066CC';
  /** 个人中心渐变起始色 */
  @State profileGradientStart: string = '#4A90D9';
  /** 个人中心渐变结束色 */
  @State profileGradientEnd: string = '#4CACFF';
  /** 头像背景色 */
  @State avatarBackground: string = '#4A90D9';
  /** 课程数量文字颜色 */
  @State courseCountText: string = '#4A90D9';
  /** 课表月份颜色 */
  @State scheduleMonth: string = '#4A90D9';
  /** 课表周次颜色 */
  @State scheduleWeek: string = '#4A90D9';
  /** 课表学期颜色 */
  @State scheduleSemester: string = '#4A90D9';
  /** 高亮颜色 */
  @State highlightColor: string = '#4A90D9';
  /** 选中的颜色目标 */
  @State selectedColorTarget: string = '';
  /** 显示颜色选择器 */
  @State showColorPicker: boolean = false;
  /** 动效手势是否启用 */
  @State motionGestureEnabled: boolean = false;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 应用上下文 */
  private context: common.UIAbilityContext | null = null;
  /** 主题颜色管理器实例 */
  private themeColorManager: ThemeColorManager = ThemeColorManager.getInstance();

  async aboutToAppear() {
    this.context = getContext(this) as common.UIAbilityContext;
    await this.themeColorManager.initialize(this.context);
    await this.loadThemeSetting();
    this.loadColorsFromManager();
    this.isLoading = false;
  }

  private loadColorsFromManager(): void {
    const colors = this.themeColorManager.getColors();
    this.headerGradientStart = colors.headerGradientStart;
    this.headerGradientEnd = colors.headerGradientEnd;
    this.profileGradientStart = colors.profileGradientStart;
    this.profileGradientEnd = colors.profileGradientEnd;
    this.avatarBackground = colors.avatarBackground;
    this.courseCountText = colors.courseCountText;
    this.scheduleMonth = colors.scheduleMonth;
    this.scheduleWeek = colors.scheduleWeek;
    this.scheduleSemester = colors.scheduleSemester;
    this.highlightColor = colors.highlightColor;
  }

  private getColorByKey(key: string): string {
    switch (key) {
      case 'headerGradientStart': return this.headerGradientStart;
      case 'headerGradientEnd': return this.headerGradientEnd;
      case 'profileGradientStart': return this.profileGradientStart;
      case 'profileGradientEnd': return this.profileGradientEnd;
      case 'avatarBackground': return this.avatarBackground;
      case 'courseCountText': return this.courseCountText;
      case 'scheduleMonth': return this.scheduleMonth;
      case 'scheduleWeek': return this.scheduleWeek;
      case 'scheduleSemester': return this.scheduleSemester;
      case 'highlightColor': return this.highlightColor;
      default: return '#4A90D9';
    }
  }

  private setColorByKey(key: string, color: string): void {
    switch (key) {
      case 'headerGradientStart': this.headerGradientStart = color; break;
      case 'headerGradientEnd': this.headerGradientEnd = color; break;
      case 'profileGradientStart': this.profileGradientStart = color; break;
      case 'profileGradientEnd': this.profileGradientEnd = color; break;
      case 'avatarBackground': this.avatarBackground = color; break;
      case 'courseCountText': this.courseCountText = color; break;
      case 'scheduleMonth': this.scheduleMonth = color; break;
      case 'scheduleWeek': this.scheduleWeek = color; break;
      case 'scheduleSemester': this.scheduleSemester = color; break;
      case 'highlightColor': this.highlightColor = color; break;
    }
  }

  private getCurrentColors(): ThemeColors {
    return {
      headerGradientStart: this.headerGradientStart,
      headerGradientEnd: this.headerGradientEnd,
      profileGradientStart: this.profileGradientStart,
      profileGradientEnd: this.profileGradientEnd,
      avatarBackground: this.avatarBackground,
      courseCountText: this.courseCountText,
      scheduleMonth: this.scheduleMonth,
      scheduleWeek: this.scheduleWeek,
      scheduleSemester: this.scheduleSemester,
      highlightColor: this.highlightColor
    };
  }

  private async loadThemeSetting(): Promise<void> {
    try {
      this.dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      const savedTheme = await this.dataPreferences.get(KEY_THEME_MODE, 'system') as ThemeMode;
      this.currentTheme = savedTheme;
      const motionGesture = await this.dataPreferences.get(KEY_MOTION_GESTURE, false) as boolean;
      this.motionGestureEnabled = motionGesture;
      AppStorage.SetOrCreate('motionGestureEnabled', motionGesture);
    } catch (error) {
      console.error('ThemeSettingsPage: Failed to load theme setting:', error);
    }
  }

  private async saveMotionGestureSetting(enabled: boolean): Promise<void> {
    try {
      if (this.dataPreferences) {
        await this.dataPreferences.put(KEY_MOTION_GESTURE, enabled);
        await this.dataPreferences.flush();
        this.motionGestureEnabled = enabled;
        AppStorage.SetOrCreate('motionGestureEnabled', enabled);
        promptAction.showToast({ message: enabled ? '智感握姿已开启' : '智感握姿已关闭' });
      }
    } catch (error) {
      console.error('ThemeSettingsPage: Failed to save motion gesture setting:', error);
      promptAction.showToast({ message: '设置失败' });
    }
  }

  private async saveThemeSetting(mode: ThemeMode): Promise<void> {
    try {
      if (this.dataPreferences && this.context) {
        await this.dataPreferences.put(KEY_THEME_MODE, mode);
        await this.dataPreferences.flush();
        this.currentTheme = mode;
        
        AppStorage.SetOrCreate('themeMode', mode);
        
        let colorMode: ConfigurationConstant.ColorMode;
        switch (mode) {
          case 'light':
            colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_LIGHT;
            break;
          case 'dark':
            colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
            break;
          default:
            colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET;
            break;
        }
        
        this.context.getApplicationContext().setColorMode(colorMode);
        promptAction.showToast({ message: '主题已切换' });
      }
    } catch (error) {
      console.error('ThemeSettingsPage: Failed to save theme setting:', error);
      promptAction.showToast({ message: '切换失败' });
    }
  }

  private async saveColorSetting(): Promise<void> {
    const colors = this.getCurrentColors();
    await this.themeColorManager.saveColors(colors);
    promptAction.showToast({ message: '颜色已保存' });
  }

  private getThemeOptions(): ThemeOption[] {
    return [
      { mode: 'light', title: '浅色模式', description: '始终使用浅色主题', icon: 'app.media.sun_min' },
      { mode: 'dark', title: '深色模式', description: '始终使用深色主题', icon: 'app.media.moon_fill' },
      { mode: 'system', title: '跟随系统', description: '根据系统设置自动切换', icon: 'app.media.circle_lefthalf_fill' }
    ];
  }

  private getColorTargets(): ColorTarget[] {
    return [
      { key: 'headerGradientStart', title: '首页头部渐变起始色', description: '欢迎界面头部背景渐变起始颜色' },
      { key: 'headerGradientEnd', title: '首页头部渐变结束色', description: '欢迎界面头部背景渐变结束颜色' },
      { key: 'profileGradientStart', title: '我的头部渐变起始色', description: '我的页面头部背景渐变起始颜色' },
      { key: 'profileGradientEnd', title: '我的头部渐变结束色', description: '我的页面头部背景渐变结束颜色' },
      { key: 'avatarBackground', title: '头像背景色', description: '个人头像圆形背景颜色' },
      { key: 'courseCountText', title: '课程计数提示词', description: '今日/明日课程数量提示文字颜色' },
      { key: 'scheduleMonth', title: '课表月份', description: '课表页面月份显示颜色' },
      { key: 'scheduleWeek', title: '课表周数', description: '课表页面周数显示颜色' },
      { key: 'scheduleSemester', title: '课表学期', description: '课表页面学期显示颜色' },
      { key: 'highlightColor', title: '高亮颜色', description: '当前周/今日高亮显示颜色' }
    ];
  }

  build() {
    Stack() {
      Column() {
        Column()
          .width('100%')
          .height(44)
        this.buildHeader()

        if (this.isLoading) {
          Column() {
            LoadingProgress()
              .width(48)
              .height(48)
              .color($r('app.color.primary'))
          }
          .width('100%')
          .layoutWeight(1)
          .justifyContent(FlexAlign.Center)
        } else {
          Scroll() {
            Column({ space: 16 }) {
              this.buildThemeModeSection()
              this.buildColorSection()
              this.buildMotionGestureSection()
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 20 })
          }
          .scrollBar(BarState.Off)
          .edgeEffect(EdgeEffect.Spring)
          .layoutWeight(1)
        }
      }
      .width('100%')
      .height('100%')
      .backgroundColor($r('app.color.background_tertiary'))

      if (this.showColorPicker && this.selectedColorTarget.length > 0) {
        this.buildColorPickerDialog()
      }
    }
    .width('100%')
    .height('100%')
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
  }

  @Builder
  buildHeader() {
    Row() {
      Image($r('app.media.chevron_left'))
        .width(24)
        .height(24)
        .fillColor($r('app.color.text_primary'))
        .onClick(() => router.back())

      Text('主题设置')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
        .textAlign(TextAlign.Center)

      Blank()
        .width(24)
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
  }

  @Builder
  buildThemeModeSection() {
    Column() {
      Text('显示模式')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')
        .padding({ left: 4, bottom: 8 })

      Column({ space: 8 }) {
        ForEach(this.getThemeOptions(), (option: ThemeOption) => {
          this.buildThemeOption(option)
        })
      }
      .width('100%')
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(12)
      .padding(4)
    }
    .width('100%')
  }

  @Builder
  buildThemeOption(option: ThemeOption) {
    Row() {
      Image($r(option.icon))
        .width(22)
        .height(22)
        .fillColor(this.currentTheme === option.mode ? this.highlightColor : $r('app.color.text_secondary'))

      Column({ space: 2 }) {
        Text(option.title)
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))

        Text(option.description)
          .fontSize(11)
          .fontColor($r('app.color.text_secondary'))
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)
      .margin({ left: 12 })

      if (this.currentTheme === option.mode) {
        Image($r('app.media.checkmark_circle'))
          .width(18)
          .height(18)
          .fillColor(this.highlightColor)
      }
    }
    .width('100%')
    .padding({ left: 12, right: 12, top: 10, bottom: 10 })
    .backgroundColor(this.currentTheme === option.mode ? this.highlightColor + '15' : 'transparent')
    .borderRadius(8)
    .onClick(() => {
      if (this.currentTheme !== option.mode) {
        this.saveThemeSetting(option.mode);
      }
    })
  }

  @Builder
  buildColorSection() {
    Column() {
      Text('主题颜色')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')
        .padding({ left: 4, bottom: 8 })

      Column() {
        ForEach(this.getColorTargets(), (target: ColorTarget, index: number) => {
          this.buildColorTargetItem(target, index === this.getColorTargets().length - 1)
        })
      }
      .width('100%')
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(12)
    }
    .width('100%')
  }

  @Builder
  buildColorTargetItem(target: ColorTarget, isLast: boolean) {
    Row() {
      Column({ space: 2 }) {
        Text(target.title)
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))

        Text(target.description)
          .fontSize(11)
          .fontColor($r('app.color.text_secondary'))
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)

      Row({ space: 8 }) {
        Stack() {
          Column()
            .width(24)
            .height(24)
            .backgroundColor(this.getColorByKey(target.key))
            .borderRadius(4)
        }
        .width(28)
        .height(28)
        .backgroundColor($r('app.color.background_tertiary'))
        .borderRadius(6)
        .alignContent(Alignment.Center)

        Image($r('app.media.chevron_right'))
          .width(14)
          .height(14)
          .fillColor($r('app.color.text_tertiary'))
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    .border({ width: 0, color: 'transparent' })
    .border({
      width: { bottom: isLast ? 0 : 0.5 },
      color: $r('app.color.divider')
    })
    .onClick(() => {
      this.selectedColorTarget = target.key;
      this.showColorPicker = true;
    })
  }

  @Builder
  buildMotionGestureSection() {
    Column() {
      Text('智感握姿')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')
        .padding({ left: 4, bottom: 8 })

      Column() {
        Row() {
          Column({ space: 2 }) {
            Text('智感握姿')
              .fontSize(15)
              .fontColor($r('app.color.text_primary'))

            Text('根据握持姿势自动调整界面布局')
              .fontSize(11)
              .fontColor($r('app.color.text_secondary'))
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)

          Toggle({
            type: ToggleType.Switch,
            isOn: this.motionGestureEnabled
          })
          .onChange((isOn: boolean) => {
            this.saveMotionGestureSetting(isOn);
          })
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      }
      .width('100%')
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(12)
    }
    .width('100%')
  }

  @Builder
  buildColorPickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Row() {
          Text('选择颜色')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))
            .layoutWeight(1)

          Image($r('app.media.xmark'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_secondary'))
            .onClick(() => {
              this.showColorPicker = false;
              this.selectedColorTarget = '';
            })
        }
        .width('100%')

        Text('预设颜色')
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')

        Flex({ wrap: FlexWrap.Wrap, justifyContent: FlexAlign.Start }) {
          ForEach(PRESET_COLORS, (color: string) => {
            Stack() {
              Column()
                .width(36)
                .height(36)
                .backgroundColor(color)
                .borderRadius(6)

              if (this.getColorByKey(this.selectedColorTarget) === color) {
                Column()
                  .width(40)
                  .height(40)
                  .borderRadius(8)
                  .border({ width: 2.5, color: color })

                Image($r('app.media.checkmark_circle'))
                  .width(16)
                  .height(16)
                  .fillColor('#FFFFFF')
              }
            }
            .width(40)
            .height(40)
            .margin({ right: 12, bottom: 12 })
            .onClick(() => {
              if (this.selectedColorTarget.length > 0) {
                this.setColorByKey(this.selectedColorTarget, color);
              }
            })
          })
        }
        .width('100%')

        Text('当前颜色')
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')

        Row({ space: 12 }) {
          Stack() {
            Column()
              .width(56)
              .height(56)
              .backgroundColor(this.getColorByKey(this.selectedColorTarget))
              .borderRadius(10)
          }
          .width(60)
          .height(60)
          .backgroundColor($r('app.color.background_tertiary'))
          .borderRadius(12)
          .alignContent(Alignment.Center)

          Column({ space: 4 }) {
            Text(this.getColorByKey(this.selectedColorTarget))
              .fontSize(14)
              .fontColor($r('app.color.text_primary'))
              .fontWeight(FontWeight.Medium)

            Text('HEX 颜色值')
              .fontSize(11)
              .fontColor($r('app.color.text_secondary'))
          }
          .alignItems(HorizontalAlign.Start)
        }
        .width('100%')

        Row({ space: 12 }) {
          Button('取消')
            .width('45%')
            .height(44)
            .fontSize(15)
            .fontColor($r('app.color.text_primary'))
            .backgroundColor($r('app.color.background_tertiary'))
            .borderRadius(22)
            .onClick(() => {
              this.showColorPicker = false;
              this.selectedColorTarget = '';
            })

          Button('确定')
            .width('45%')
            .height(44)
            .fontSize(15)
            .fontColor('#FFFFFF')
            .backgroundColor(this.highlightColor)
            .borderRadius(22)
            .onClick(() => {
              this.saveColorSetting();
              this.showColorPicker = false;
              this.selectedColorTarget = '';
            })
        }
        .width('100%')
        .justifyContent(FlexAlign.Center)
      }
      .width('85%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(16)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor('rgba(0,0,0,0.5)')
    .onClick(() => {
      this.showColorPicker = false;
      this.selectedColorTarget = '';
    })
  }
}

export { ThemeSettingsPage };
