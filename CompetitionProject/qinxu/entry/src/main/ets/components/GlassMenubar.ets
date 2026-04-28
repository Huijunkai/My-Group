/**
 * 沉静光感底部导航栏组件模块
 *
 * 对齐鸿蒙6.1沉静光感设计语言：
 * - 沉静毛玻璃：低模糊度 + 材质感半透明背景
 * - 微光高光线：模拟自然光照射的细腻顶部光线
 * - 弥散阴影：大半径低透明度柔和投影，取代锐利阴影
 * - 光感指示器：选中态底部光点
 * - 沉静动效：更柔和的弹簧曲线，优雅过渡
 * - 按压反馈 + 暗色模式自适应
 *
 * @module GlassMenubar
 */

import { HomePage } from '../pages/HomePage';
import { SchedulePage } from '../pages/SchedulePage';
import { ProfilePage } from '../pages/ProfilePage';
import { common, ConfigurationConstant } from '@kit.AbilityKit';
import { curves } from '@kit.ArkUI';

// ==================== 尺寸常量 ====================
const NAVBAR_HEIGHT = 58;
const NAVBAR_WIDTH_PERCENT = '92%';
const NAVBAR_BORDER_RADIUS = 28;
const NAVBAR_POSITION_Y = '91%';

const ICON_SIZE_SELECTED = 24;
const ICON_SIZE_UNSELECTED = 22;
const ICON_TEXT_GAP = 3;
const TEXT_SIZE = 10;

// 微光高光线
const HIGHLIGHT_LINE_HEIGHT = 0.5;

// 光感指示器（底部光点）
const INDICATOR_DOT_SIZE = 4;
const INDICATOR_DOT_BOTTOM_MARGIN = 2;

// ==================== 动画常量 ====================
// 沉静动效：更慢、更优雅的过渡
const ICON_SCALE_DURATION = 350;
const TEXT_OPACITY_DURATION = 300;
const INDICATOR_DURATION = 300;
const PRESS_FEEDBACK_DURATION = 200;

const PRESS_SCALE = 0.94;

// 沉静弹簧曲线：降低刚度，增加阻尼，减少弹跳感
const SPRING_VELOCITY = 0;
const SPRING_MASS = 1;
const SPRING_STIFFNESS = 280;
const SPRING_DAMPING = 38;

// 按压反馈弹簧：稍硬一些以保持响应感
const PRESS_SPRING_STIFFNESS = 500;
const PRESS_SPRING_DAMPING = 35;

// 毛玻璃模糊半径（沉静感：降低模糊强度）
const BACKDROP_BLUR_RADIUS = 28;

// ==================== Tab项配置接口 ====================
interface TabItemConfig {
  iconUnselected: string;
  iconSelected: string;
  text: string;
  accessibilityText: string;
}

// ==================== 颜色方案接口 ====================
interface MenubarColorSchemeType {
  // 导航栏背景
  background: string;
  // 微光高光线（顶部）
  highlightLine: string;
  // 边框
  border: string;
  // 弥散外层阴影
  shadowOuterColor: string;
  // 内层柔和阴影
  shadowInnerColor: string;
  // 选中态图标色
  iconSelected: ResourceStr;
  // 未选中态图标色
  iconUnselected: string;
  // 选中态文字色
  textSelected: ResourceStr;
  // 未选中态文字色
  textUnselected: string;
  // 光感指示器色（底部光点）
  indicatorColor: ResourceStr;
}

// ==================== 亮色/暗色颜色方案 ====================
// 鸿蒙6.1沉静光感：更低饱和度、更柔和的过渡
const LIGHT_SCHEME: MenubarColorSchemeType = {
  // 沉静毛玻璃：更高透明度，让底层内容更自然透出
  background: 'rgba(255, 255, 255, 0.65)',
  // 微光高光线：极细腻的白色光线，模拟自然光
  highlightLine: 'rgba(255, 255, 255, 0.45)',
  // 极淡边框
  border: 'rgba(0, 0, 0, 0.03)',
  // 弥散外层阴影：大半径、低透明度
  shadowOuterColor: 'rgba(0, 0, 0, 0.06)',
  // 内层柔和阴影
  shadowInnerColor: 'rgba(0, 0, 0, 0.03)',
  // 选中态：主题色
  iconSelected: $r('app.color.primary'),
  // 未选中态：沉静灰
  iconUnselected: 'rgba(0, 0, 0, 0.35)',
  // 选中态文字
  textSelected: $r('app.color.primary'),
  // 未选中态文字
  textUnselected: 'rgba(0, 0, 0, 0.35)',
  // 光感指示器
  indicatorColor: $r('app.color.primary'),
};

const DARK_SCHEME: MenubarColorSchemeType = {
  // 暗色沉静毛玻璃
  background: 'rgba(28, 28, 35, 0.68)',
  // 暗色微光高光线
  highlightLine: 'rgba(255, 255, 255, 0.08)',
  // 暗色边框
  border: 'rgba(255, 255, 255, 0.06)',
  // 暗色弝散阴影
  shadowOuterColor: 'rgba(0, 0, 0, 0.20)',
  // 暗色内层阴影
  shadowInnerColor: 'rgba(80, 80, 120, 0.10)',
  // 暗色选中态
  iconSelected: $r('app.color.primary'),
  // 暗色未选中态
  iconUnselected: 'rgba(255, 255, 255, 0.35)',
  // 暗色选中文字
  textSelected: $r('app.color.primary'),
  // 暗色未选中文字
  textUnselected: 'rgba(255, 255, 255, 0.35)',
  // 暗色光感指示器
  indicatorColor: $r('app.color.primary'),
};

// ==================== Tab项配置数据 ====================
const TAB_ITEMS: TabItemConfig[] = [
  {
    iconUnselected: 'app.media.house',
    iconSelected: 'app.media.house_fill',
    text: '首页',
    accessibilityText: '首页标签页',
  },
  {
    iconUnselected: 'app.media.calendar',
    iconSelected: 'app.media.calendar_fill',
    text: '课表',
    accessibilityText: '课表标签页',
  },
  {
    iconUnselected: 'app.media.person_2',
    iconSelected: 'app.media.person_2_fill',
    text: '我的',
    accessibilityText: '我的标签页',
  },
];

/**
 * 沉静光感底部导航栏组件
 *
 * 对齐鸿蒙6.1沉静光感设计语言：
 * - 沉静毛玻璃背景 + 微光高光线 + 弥散双层阴影
 * - 光感指示器（底部光点）
 * - 图标缩放过渡(22↔24vp) + 沉静弹簧曲线动效
 * - 按压反馈(0.94缩放回弹) + 暗色模式自适应
 *
 * @example
 * ```typescript
 * GlassMenubar()
 * ```
 */
@Component
export struct GlassMenubar {
  /** 当前选中的Tab索引 */
  @State currentIndex: number = 0;
  /** 暗色模式状态（双向绑定） */
  @StorageLink('isDarkMode') isDarkMode: boolean = false;

  /** 各Tab图标尺寸（选中24vp，未选中22vp） */
  @State iconScales: number[] = [ICON_SIZE_SELECTED, ICON_SIZE_UNSELECTED, ICON_SIZE_UNSELECTED];
  /** 各Tab文字透明度（选中1.0，未选中0.35） */
  @State textOpacities: number[] = [1.0, 0.35, 0.35];
  /** 各Tab按压缩放比例（按下0.94，松开1.0） */
  @State pressScales: number[] = [1.0, 1.0, 1.0];
  /** 各Tab光感指示器透明度（选中1.0，未选中0.0） */
  @State indicatorOpacities: number[] = [1.0, 0.0, 0.0];

  /** Swiper控制器 */
  private swiperController: SwiperController = new SwiperController();

  aboutToAppear(): void {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const isDark = context.config.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
      AppStorage.SetOrCreate('isDarkMode', isDark);
    } catch (e) {
      console.warn('GlassMenubar: Failed to init dark mode:', JSON.stringify(e));
    }
  }

  /** 获取当前颜色方案 */
  private getColorScheme(): MenubarColorSchemeType {
    return this.isDarkMode ? DARK_SCHEME : LIGHT_SCHEME;
  }

  /** 沉静弹簧曲线 */
  private getSpringCurve(): ICurve {
    return curves.springCurve(SPRING_VELOCITY, SPRING_MASS, SPRING_STIFFNESS, SPRING_DAMPING);
  }

  /** 按压反馈弹簧曲线 */
  private getPressSpringCurve(): ICurve {
    return curves.springCurve(SPRING_VELOCITY, SPRING_MASS, PRESS_SPRING_STIFFNESS, PRESS_SPRING_DAMPING);
  }

  /** Tab点击切换逻辑 */
  private onTabClick(targetIndex: number): void {
    if (targetIndex === this.currentIndex) {
      return;
    }

    this.currentIndex = targetIndex;

    // 图标缩放动画
    animateTo({
      duration: ICON_SCALE_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newScales: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newScales.push(i === targetIndex ? ICON_SIZE_SELECTED : ICON_SIZE_UNSELECTED);
      }
      this.iconScales = newScales;
    });

    // 文字透明度动画
    animateTo({
      duration: TEXT_OPACITY_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newOpacities: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newOpacities.push(i === targetIndex ? 1.0 : 0.35);
      }
      this.textOpacities = newOpacities;
    });

    // 光感指示器动画
    animateTo({
      duration: INDICATOR_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newIndicatorOpacities: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newIndicatorOpacities.push(i === targetIndex ? 1.0 : 0.0);
      }
      this.indicatorOpacities = newIndicatorOpacities;
    });

    // 通知Swiper切换页面
    this.swiperController.changeIndex(targetIndex);
  }

  /** 按压反馈逻辑 */
  private onTabTouch(index: number, event: TouchEvent): void {
    if (event.type === TouchType.Down) {
      animateTo({
        duration: PRESS_FEEDBACK_DURATION,
        curve: this.getPressSpringCurve()
      }, () => {
        const newScales = [...this.pressScales];
        newScales[index] = PRESS_SCALE;
        this.pressScales = newScales;
      });
    } else if (event.type === TouchType.Up) {
      animateTo({
        duration: PRESS_FEEDBACK_DURATION,
        curve: this.getPressSpringCurve()
      }, () => {
        const newScales = [...this.pressScales];
        newScales[index] = 1.0;
        this.pressScales = newScales;
      });
      this.onTabClick(index);
    } else if (event.type === TouchType.Cancel) {
      animateTo({
        duration: PRESS_FEEDBACK_DURATION,
        curve: this.getPressSpringCurve()
      }, () => {
        const newScales = [...this.pressScales];
        newScales[index] = 1.0;
        this.pressScales = newScales;
      });
    }
  }

  /** Swiper滑动同步逻辑 */
  private onSwiperChange(newIndex: number): void {
    if (newIndex === this.currentIndex) {
      return;
    }

    this.currentIndex = newIndex;

    animateTo({
      duration: ICON_SCALE_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newScales: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newScales.push(i === newIndex ? ICON_SIZE_SELECTED : ICON_SIZE_UNSELECTED);
      }
      this.iconScales = newScales;
    });

    animateTo({
      duration: TEXT_OPACITY_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newOpacities: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newOpacities.push(i === newIndex ? 1.0 : 0.35);
      }
      this.textOpacities = newOpacities;
    });

    animateTo({
      duration: INDICATOR_DURATION,
      curve: this.getSpringCurve()
    }, () => {
      const newIndicatorOpacities: number[] = [];
      for (let i = 0; i < TAB_ITEMS.length; i++) {
        newIndicatorOpacities.push(i === newIndex ? 1.0 : 0.0);
      }
      this.indicatorOpacities = newIndicatorOpacities;
    });
  }

  build() {
    Stack() {
      // 页面内容区域
      Swiper(this.swiperController) {
        HomePage()
        SchedulePage({})
        ProfilePage()
      }
      .index(this.currentIndex)
      .autoPlay(false)
      .indicator(false)
      .loop(false)
      .duration(350)
      .curve(this.getSpringCurve())
      .onChange((index: number) => {
        this.onSwiperChange(index);
      })
      .width('100%')
      .height('100%')

      // 沉静光感底部导航栏
      Column() {
        // 微光高光线：模拟自然光照射的细腻顶部光线
        Row()
          .width('60%')
          .height(HIGHLIGHT_LINE_HEIGHT)
          .backgroundColor(this.getColorScheme().highlightLine)
          .blur(0.5)

        // 导航栏内容区
        Row() {
          ForEach(TAB_ITEMS, (item: TabItemConfig, index?: number) => {
            if (index !== undefined) {
              this.buildTabItem(item, index)
            }
          }, (item: TabItemConfig, index?: number) => `${index}`)
        }
        .layoutWeight(1)
        .justifyContent(FlexAlign.SpaceEvenly)
      }
      .width(NAVBAR_WIDTH_PERCENT)
      .height(NAVBAR_HEIGHT)
      .position({ x: '4%', y: NAVBAR_POSITION_Y })
      .borderRadius(NAVBAR_BORDER_RADIUS)
      // 弥散双层阴影：大半径低透明度，取代锐利投影
      .shadow({ radius: 32, color: this.getColorScheme().shadowOuterColor, offsetX: 0, offsetY: 8 })
      .shadow({ radius: 12, color: this.getColorScheme().shadowInnerColor, offsetX: 0, offsetY: 2 })
      .backgroundColor(this.getColorScheme().background)
      // 沉静毛玻璃：降低模糊强度，保留材质感
      .backdropBlur(BACKDROP_BLUR_RADIUS)
      .border({ width: 0.5, color: this.getColorScheme().border, style: BorderStyle.Solid })
    }
    .width('100%')
    .height('100%')
  }

  /**
   * 构建导航Tab项
   *
   * 选中态：填充图标(24vp) + 主题色 + 光感指示器
   * 未选中态：线性图标(22vp) + 沉静灰 + 文字透明度0.35
   *
   * @param item - Tab项配置
   * @param index - Tab项索引
   */
  @Builder
  buildTabItem(item: TabItemConfig, index: number) {
    Column({ space: ICON_TEXT_GAP }) {
      // 图标
      Image($r(this.currentIndex === index ? item.iconSelected : item.iconUnselected))
        .width(this.iconScales[index])
        .height(this.iconScales[index])
        .fillColor(this.currentIndex === index
          ? this.getColorScheme().iconSelected
          : this.getColorScheme().iconUnselected)
        .animation({
          duration: ICON_SCALE_DURATION,
          curve: this.getSpringCurve()
        })

      // 文字标签
      Text(item.text)
        .fontSize(TEXT_SIZE)
        .fontColor(this.currentIndex === index
          ? this.getColorScheme().textSelected
          : this.getColorScheme().textUnselected)
        .opacity(this.textOpacities[index])
        .fontWeight(this.currentIndex === index ? FontWeight.Medium : FontWeight.Normal)
        .animation({
          duration: TEXT_OPACITY_DURATION,
          curve: this.getSpringCurve()
        })

      // 光感指示器（底部光点）
      Row()
        .width(INDICATOR_DOT_SIZE)
        .height(INDICATOR_DOT_SIZE)
        .borderRadius(INDICATOR_DOT_SIZE / 2)
        .backgroundColor(this.getColorScheme().indicatorColor)
        .opacity(this.indicatorOpacities[index])
        .margin({ top: INDICATOR_DOT_BOTTOM_MARGIN })
        .animation({
          duration: INDICATOR_DURATION,
          curve: this.getSpringCurve()
        })
    }
    .layoutWeight(1)
    .scale({ x: this.pressScales[index], y: this.pressScales[index] })
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center)
    .accessibilityText(item.accessibilityText)
    .accessibilityDescription(`${item.text}标签页${this.currentIndex === index ? '，已选中' : ''}`)
    .onTouch((event: TouchEvent) => {
      this.onTabTouch(index, event);
    })
  }
}
