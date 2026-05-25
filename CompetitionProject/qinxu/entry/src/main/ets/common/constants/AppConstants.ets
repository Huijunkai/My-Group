/**
 * 应用常量配置模块
 * 
 * 定义应用级别的常量配置，包括：
 * - 应用基本信息
 * - 偏好设置键名
 * - 时间相关常量
 * - 布局相关常量
 * 
 * @module AppConstants
 */

/**
 * 应用常量类
 * 
 * 提供统一的应用配置管理
 * 
 * @example
 * ```typescript
 * const appName = AppConstants.APP_NAME;
 * const padding = AppConstants.LAYOUT.DEFAULT_PADDING;
 * ```
 */
export class AppConstants {
  /** 应用名称 */
  static readonly APP_NAME: string = 'NNLG';
  
  /** 应用版本号 */
  static readonly VERSION: string = '1.0.0';
  
  /**
   * 偏好设置键名常量
   * 
   * @property THEME - 主题偏好设置键
   * @property THEME_MODE - 主题模式键
   * @property USER_DATA - 用户数据偏好设置键
   */
  static readonly PREFERENCES = {
    THEME: 'theme_preferences',
    THEME_MODE: 'themeMode',
    USER_DATA: 'user_preferences'
  } as const;
  
  /**
   * 时间相关常量（单位：毫秒）
   * 
   * @property ANIMATION_DURATION_SHORT - 短动画时长（200ms）
   * @property ANIMATION_DURATION_MEDIUM - 中等动画时长（300ms）
   * @property ANIMATION_DURATION_LONG - 长动画时长（500ms）
   * @property DEBOUNCE_DELAY - 防抖延迟时间（300ms）
   * @property THROTTLE_DELAY - 节流延迟时间（100ms）
   */
  static readonly TIME = {
    ANIMATION_DURATION_SHORT: 200,
    ANIMATION_DURATION_MEDIUM: 300,
    ANIMATION_DURATION_LONG: 500,
    DEBOUNCE_DELAY: 300,
    THROTTLE_DELAY: 100
  } as const;
  
  /**
   * 布局相关常量（单位：vp）
   * 
   * @property DEFAULT_PADDING - 默认内边距（16vp）
   * @property SMALL_PADDING - 小内边距（8vp）
   * @property LARGE_PADDING - 大内边距（24vp）
   * @property DEFAULT_BORDER_RADIUS - 默认圆角（12vp）
   * @property SMALL_BORDER_RADIUS - 小圆角（8vp）
   */
  static readonly LAYOUT = {
    DEFAULT_PADDING: 16,
    SMALL_PADDING: 8,
    LARGE_PADDING: 24,
    DEFAULT_BORDER_RADIUS: 12,
    SMALL_BORDER_RADIUS: 8
  } as const;
}
