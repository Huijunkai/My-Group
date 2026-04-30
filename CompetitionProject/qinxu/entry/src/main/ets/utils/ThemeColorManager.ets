/**
 * 主题颜色管理模块
 * 
 * 提供主题颜色管理功能，包括：
 * - 主题颜色存储
 * - 预设颜色管理
 * - 颜色持久化
 * 
 * @module ThemeColorManager
 */

import { preferences } from '@kit.ArkData';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'theme_color_preferences';

/**
 * 主题颜色接口
 */
interface ThemeColors {
  /** 头部渐变起始色 */
  headerGradientStart: string;
  /** 头部渐变结束色 */
  headerGradientEnd: string;
  /** 个人中心渐变起始色 */
  profileGradientStart: string;
  /** 个人中心渐变结束色 */
  profileGradientEnd: string;
  /** 头像背景色 */
  avatarBackground: string;
  /** 课程数量文字颜色 */
  courseCountText: string;
  /** 课表月份颜色 */
  scheduleMonth: string;
  /** 课表周次颜色 */
  scheduleWeek: string;
  /** 课表学期颜色 */
  scheduleSemester: string;
  /** 高亮颜色 */
  highlightColor: string;
}

/**
 * 默认颜色配置
 */
const DEFAULT_COLORS: ThemeColors = {
  headerGradientStart: '#4A90D9',
  headerGradientEnd: '#0066CC',
  profileGradientStart: '#4A90D9',
  profileGradientEnd: '#4CACFF',
  avatarBackground: '#4A90D9',
  courseCountText: '#4A90D9',
  scheduleMonth: '#4A90D9',
  scheduleWeek: '#4A90D9',
  scheduleSemester: '#4A90D9',
  highlightColor: '#4A90D9'
};

/** 预设颜色列表 */
const PRESET_COLORS: string[] = [
  '#4A90D9',
  '#34C759',
  '#FF9500',
  '#FF3B30',
  '#AF52DE',
  '#FF2D55',
  '#00C7BE',
  '#5856D6',
  '#FF6482',
  '#64D2FF'
];

/**
 * 复制主题颜色
 * 
 * @param source - 源颜色对象
 * @returns 颜色副本
 */
function copyThemeColors(source: ThemeColors): ThemeColors {
  return {
    headerGradientStart: source.headerGradientStart,
    headerGradientEnd: source.headerGradientEnd,
    profileGradientStart: source.profileGradientStart,
    profileGradientEnd: source.profileGradientEnd,
    avatarBackground: source.avatarBackground,
    courseCountText: source.courseCountText,
    scheduleMonth: source.scheduleMonth,
    scheduleWeek: source.scheduleWeek,
    scheduleSemester: source.scheduleSemester,
    highlightColor: source.highlightColor
  };
}

/**
 * 复制预设颜色数组
 * 
 * @param source - 源颜色数组
 * @returns 颜色数组副本
 */
function copyPresetColors(source: string[]): string[] {
  const result: string[] = [];
  for (let i = 0; i < source.length; i++) {
    result.push(source[i]);
  }
  return result;
}

/**
 * 主题颜色管理类
 * 
 * 管理应用的主题颜色设置
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const manager = ThemeColorManager.getInstance();
 * await manager.initialize(context);
 * const colors = manager.getColors();
 * ```
 */
class ThemeColorManager {
  /** 单例实例 */
  private static instance: ThemeColorManager;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 当前颜色配置 */
  private colors: ThemeColors = copyThemeColors(DEFAULT_COLORS);

  static getInstance(): ThemeColorManager {
    if (!ThemeColorManager.instance) {
      ThemeColorManager.instance = new ThemeColorManager();
    }
    return ThemeColorManager.instance;
  }

  async initialize(context: Context): Promise<void> {
    try {
      this.dataPreferences = await preferences.getPreferences(context, PREFERENCES_NAME);
      await this.loadColors();
      AppStorage.SetOrCreate('themeColors', this.colors);
    } catch (error) {
      console.error('ThemeColorManager: Failed to initialize:', error);
    }
  }

  private async loadColors(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const savedColors = await this.dataPreferences.get('themeColors', '') as string;
      if (savedColors) {
        this.colors = JSON.parse(savedColors) as ThemeColors;
      }
    } catch (error) {
      console.error('ThemeColorManager: Failed to load colors:', error);
    }
  }

  async saveColors(colors: ThemeColors): Promise<void> {
    this.colors = copyThemeColors(colors);
    
    if (this.dataPreferences) {
      await this.dataPreferences.put('themeColors', JSON.stringify(colors));
      await this.dataPreferences.flush();
    }
    
    AppStorage.SetOrCreate('themeColors', colors);
  }

  getColors(): ThemeColors {
    return copyThemeColors(this.colors);
  }

  getPresets(): string[] {
    return copyPresetColors(PRESET_COLORS);
  }

  getDefaultColors(): ThemeColors {
    return copyThemeColors(DEFAULT_COLORS);
  }
}

export { ThemeColorManager, ThemeColors, PRESET_COLORS, DEFAULT_COLORS };
