/**
 * 课程提醒服务模块
 * 
 * 功能说明：
 * 1. 管理课程提醒的创建、更新和删除
 * 2. 支持课前提醒和明日课程提醒两种类型
 * 3. 自动清理过期的提醒
 * 4. 使用系统代理通知服务（reminderAgentManager）
 * 
 * 提醒类型：
 * - 课前提醒：在课程开始前指定时间发送提醒
 * - 明日课程提醒：在夜间指定时间提醒明天的课程安排
 * 
 * @module CourseReminderService
 */

import { reminderAgentManager } from '@kit.BackgroundTasksKit';
import { notificationManager } from '@kit.NotificationKit';
import { preferences } from '@kit.ArkData';
import { wantAgent } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { SettingsData } from '../model/SettingsData';
import { BusinessError } from '@kit.BasicServicesKit';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'course_reminder_settings';

/** 日志域 */
const DOMAIN = 0x0000;

/**
 * 课程提醒设置接口
 * 
 * @property enabled - 是否启用课程提醒
 * @property advanceMinutes - 提前提醒时间（分钟）
 * @property remindBeforeClass - 是否启用课前提醒
 * @property remindTomorrowCourse - 是否启用明日课程提醒
 * @property nightReminderTime - 夜间提醒时间（格式：HH:mm）
 */
export interface CourseReminderSettings {
  enabled: boolean;
  advanceMinutes: number;
  remindBeforeClass: boolean;
  remindTomorrowCourse: boolean;
  nightReminderTime: string;
}

/**
 * 课程提醒信息接口
 * 
 * 用于描述一门课程的基本信息，供创建提醒时使用
 * 
 * @property courseName - 课程名称
 * @property location - 上课地点
 * @property teacher - 授课教师
 * @property dayOfWeek - 星期几（1-7，1=周一）
 * @property startTime - 开始时间（格式：HH:mm）
 * @property endTime - 结束时间（格式：HH:mm）
 * @property week - 周次
 * @property courseType - 课程类型（可选）
 * @property weeks - 上课周次数组（可选）
 */
export interface CourseReminderInfo {
  courseName: string;
  location: string;
  teacher: string;
  dayOfWeek: number;
  startTime: string;
  endTime: string;
  week: number;
  courseType?: string;
  weeks?: number[];
}

/**
 * 提醒记录接口
 * 
 * 用于存储在本地偏好设置中的提醒记录
 * 
 * @property reminderId - 系统分配的提醒ID
 * @property courseKey - 课程唯一标识键（格式：week_dayOfWeek_startTime_courseName）
 * @property createdAt - 创建时间戳（毫秒）
 * @property reminderType - 提醒类型：timer（计时器）或 calendar（日历）
 * @property triggerTime - 触发时间戳（秒），可选
 * @property courseName - 课程名称（可选，用于调试）
 * @property location - 上课地点（可选）
 * @property teacher - 授课教师（可选）
 */
interface ReminderRecord {
  reminderId: number;
  courseKey: string;
  createdAt: number;
  reminderType: 'timer' | 'calendar';
  triggerTime?: number;
  courseName?: string;
  location?: string;
  teacher?: string;
}

/**
 * 刷新状态接口
 * 
 * 用于控制提醒刷新频率，避免过于频繁的刷新
 * 
 * @property lastRefreshTime - 上次刷新时间戳（毫秒）
 * @property lastRefreshWeek - 上次刷新时的周次
 */
interface RefreshState {
  lastRefreshTime: number;
  lastRefreshWeek: number;
}

/**
 * 触发时间计算结果接口
 * 
 * 统一封装触发时间计算结果，确保过期判断和提醒发布使用同一时间源
 * 
 * @property triggerTimeSeconds - 触发时间戳（秒）
 * @property isExpired - 是否已过期
 * @property triggerDate - 触发时间 Date 对象
 */
interface TriggerTimeResult {
  triggerTimeSeconds: number;
  isExpired: boolean;
  triggerDate: Date;
}

/**
 * 即将触发的提醒接口
 * 
 * 用于展示即将触发的提醒列表
 * 
 * @property courseName - 课程名称
 * @property location - 上课地点
 * @property teacher - 授课教师
 * @property triggerTime - 触发时间
 * @property type - 提醒类型：before_class（课前）、morning（早晨）、tomorrow（明日）
 */
interface UpcomingReminder {
  courseName: string;
  location: string;
  teacher: string;
  triggerTime: Date;
  type: 'before_class' | 'morning' | 'tomorrow';
}

/**
 * 提醒统计摘要接口
 * 
 * 用于展示当前提醒的统计信息
 * 
 * @property total - 提醒总数
 * @property beforeClass - 课前提醒数量
 * @property morning - 早晨提醒数量
 * @property tomorrow - 明日提醒数量
 * @property remaining - 剩余可用容量
 */
interface RemindersSummary {
  total: number;
  beforeClass: number;
  morning: number;
  tomorrow: number;
  remaining: number;
}

/**
 * 课程提醒服务类
 * 
 * 单例模式实现，提供课程提醒的完整管理功能
 * 
 * 主要功能：
 * 1. 创建和管理课程提醒
 * 2. 自动清理过期提醒
 * 3. 支持批量设置提醒
 * 4. 提供提醒统计信息
 * 
 * 使用示例：
 * ```typescript
 * const service = CourseReminderService.getInstance();
 * await service.initialize(context);
 * await service.publishCourseReminder(courseInfo);
 * ```
 */
class CourseReminderService {
  /** 单例实例 */
  private static instance: CourseReminderService;
  
  /** 偏好设置存储对象 */
  private dataPreferences: preferences.Preferences | null = null;
  
  /** 当前提醒设置 */
  private settings: CourseReminderSettings = {
    enabled: false,
    advanceMinutes: 15,
    remindBeforeClass: true,
    remindTomorrowCourse: true,
    nightReminderTime: '21:00'
  };
  
  /** 活跃提醒记录映射表（key: courseKey, value: ReminderRecord） */
  private activeReminders: Map<string, ReminderRecord> = new Map();
  
  /** 刷新状态记录 */
  private refreshState: RefreshState = {
    lastRefreshTime: 0,
    lastRefreshWeek: 0
  };
  
  /** 应用上下文 */
  private context: Context | null = null;
  
  /** 提醒数量上限（系统限制） */
  private maxRemindersLimit: number = 30;
  
  /** 最小刷新间隔（1小时，毫秒） */
  private readonly MIN_REFRESH_INTERVAL: number = 60 * 60 * 1000;

  /**
   * 获取单例实例
   * 
   * @returns CourseReminderService 实例
   */
  static getInstance(): CourseReminderService {
    if (!CourseReminderService.instance) {
      CourseReminderService.instance = new CourseReminderService();
    }
    return CourseReminderService.instance;
  }

  /**
   * 初始化服务
   * 
   * 必须在使用其他方法前调用此方法
   * 
   * @param context - 应用上下文
   */
  async initialize(context: Context): Promise<void> {
    this.context = context;
    try {
      this.dataPreferences = await preferences.getPreferences(context, PREFERENCES_NAME);
      await this.loadSettings();
      await this.loadActiveReminders();
      await this.setupNotificationSlots();
      this.setupNotificationCancelListener();
      console.info('CourseReminderService: Initialized successfully');
    } catch (error) {
      console.error('CourseReminderService: Failed to initialize:', error);
    }
  }

  /**
   * 设置通知槽
   * 
   * 配置通知的显示方式（震动、角标等）
   */
  private async setupNotificationSlots(): Promise<void> {
    try {
      const contentInfoSlot: notificationManager.NotificationSlot = {
        type: 1,
        level: 4,
        enabled: true,
        badgeFlag: true,
        vibrationEnabled: true,
        lightEnabled: false
      };

      await reminderAgentManager.addNotificationSlot(contentInfoSlot);
      console.info('CourseReminderService: Notification slot added successfully');
    } catch (error) {
      console.error('CourseReminderService: Failed to add notification slot:', error);
    }
  }

  /**
   * 设置通知取消监听器
   * 
   * 由于 notificationManager.on('cancel') 在当前 SDK 不可用，
   * 改用前台切换时通过 getValidReminders 同步清理已删除的提醒记录。
   * 此方法为空操作，实际同步逻辑在 syncActiveRemindersWithSystem 中实现。
   */
  private setupNotificationCancelListener(): void {
    console.info('CourseReminderService: Notification cancel sync will be handled via getValidReminders on foreground');
  }

  /**
   * 移除通知取消监听器
   * 
   * 保留为空操作，保持接口兼容
   */
  removeNotificationCancelListener(): void {
    console.info('CourseReminderService: Notification cancel listener removed (no-op)');
  }

  /**
   * 与系统有效提醒同步清理本地记录
   * 
   * 在应用前台切换时调用，通过 reminderAgentManager.getValidReminders() 
   * 获取系统中仍有效的提醒ID列表，将本地 activeReminders 中
   * 不在有效列表中的记录清理掉（用户已在通知栏删除的提醒）
   */
  private async syncActiveRemindersWithSystem(): Promise<void> {
    try {
      const validReminders = await reminderAgentManager.getValidReminders();
      const validNotificationIds = new Set<number>();
      
      for (let i = 0; i < validReminders.length; i++) {
        const reminder = validReminders[i];
        if (reminder.notificationId !== undefined) {
          validNotificationIds.add(reminder.notificationId);
        }
      }
      
      console.info('CourseReminderService: System valid reminders count:', validReminders.length,
        'local activeReminders count:', this.activeReminders.size);
      
      // 通过 notificationId 匹配：计算每条本地记录的 notificationId，与系统有效列表比对
      const staleKeys: string[] = [];
      this.activeReminders.forEach((record: ReminderRecord, key: string) => {
        const localNotificationId = this.computeNotificationIdFromKey(key);
        if (localNotificationId >= 0 && !validNotificationIds.has(localNotificationId)) {
          staleKeys.push(key);
        }
      });
      
      if (staleKeys.length > 0) {
        for (const key of staleKeys) {
          this.activeReminders.delete(key);
          console.info('CourseReminderService: Removed stale reminder record (no longer in system):', key);
        }
        await this.saveActiveReminders();
        console.info('CourseReminderService: Synced with system, removed', staleKeys.length, 'stale records');
      }
    } catch (error) {
      console.error('CourseReminderService: Failed to sync active reminders with system:', error);
    }
  }

  /**
   * 根据提醒记录的 courseKey 计算其 notificationId
   * 
   * 用于与系统有效提醒列表中的 notificationId 匹配
   * 
   * @param key - 提醒记录的 courseKey
   * @returns notificationId，无法计算时返回 -1
   */
  private computeNotificationIdFromKey(key: string): number {
    // 明日提醒：key 以 tomorrow_ 开头，notificationId = 2000000 + dayOfWeek
    if (key.startsWith('tomorrow_')) {
      const dayOfWeek = parseInt(key.split('_')[1]) || 1;
      return 2000000 + dayOfWeek;
    }
    
    // 早晨提醒：key 以 morning_ 开头，notificationId = 3000000 + week*10 + dayOfWeek
    if (key.startsWith('morning_')) {
      const parts = key.split('_');
      const week = parseInt(parts[1]) || 1;
      const dayOfWeek = parseInt(parts[2]) || 1;
      return 3000000 + week * 10 + dayOfWeek;
    }
    
    // 课前提醒：courseKey 格式为 week_dayOfWeek_startTime_courseName
    // 使用与 generateNotificationId 相同的哈希算法
    const parts = key.split('_');
    if (parts.length >= 4) {
      const week = parts[0];
      const dayOfWeek = parts[1];
      const startTime = parts[2];
      const courseName = parts.slice(3).join('_');
      const courseInfo = `${courseName}_${week}_${dayOfWeek}_${startTime}`;
      let hash = 0;
      for (let i = 0; i < courseInfo.length; i++) {
        const char = courseInfo.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
      }
      return Math.abs(hash) % 1000000;
    }
    
    return -1;
  }

  /**
   * 创建用于跳转到主页的 WantAgent
   * 
   * 当用户点击通知时，会跳转到应用主页
   * 
   * @returns WantAgent 对象，失败返回 null
   */
  private async createWantAgentForHomePage(): Promise<ESObject | null> {
    if (!this.context) {
      hilog.error(DOMAIN, 'CourseReminderService', 'Context is null, cannot create WantAgent');
      return null;
    }

    try {
      const wantAgentInfo: wantAgent.WantAgentInfo = {
        wants: [
          {
            bundleName: 'com.example.qinxu',
            abilityName: 'EntryAbility',
            parameters: {
              'reminderSource': 'course_reminder',
              'targetPage': 'HomePage'
            }
          }
        ],
        operationType: wantAgent.OperationType.START_ABILITY,
        requestCode: 1001,
        wantAgentFlags: [wantAgent.WantAgentFlags.CONSTANT_FLAG]
      };

      const agent = await wantAgent.getWantAgent(wantAgentInfo);
      hilog.info(DOMAIN, 'CourseReminderService', 'WantAgent created successfully');
      return agent;
    } catch (error) {
      hilog.error(DOMAIN, 'CourseReminderService', 'Error creating WantAgent: %{public}s', JSON.stringify(error));
      return null;
    }
  }

  /**
   * 从本地存储加载设置
   */
  private async loadSettings(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const savedSettings = await this.dataPreferences.get('settings', '') as string;
      if (savedSettings) {
        const parsed: CourseReminderSettings = JSON.parse(savedSettings);
        if (parsed.enabled !== undefined) this.settings.enabled = parsed.enabled;
        if (parsed.advanceMinutes !== undefined) this.settings.advanceMinutes = parsed.advanceMinutes;
        if (parsed.remindBeforeClass !== undefined) this.settings.remindBeforeClass = parsed.remindBeforeClass;
        if (parsed.remindTomorrowCourse !== undefined) this.settings.remindTomorrowCourse = parsed.remindTomorrowCourse;
        if (parsed.nightReminderTime !== undefined) this.settings.nightReminderTime = parsed.nightReminderTime;
      }
    } catch (error) {
      console.error('CourseReminderService: Failed to load settings:', error);
    }
  }

  /**
   * 保存设置到本地存储
   */
  private async saveSettings(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      await this.dataPreferences.put('settings', JSON.stringify(this.settings));
      await this.dataPreferences.flush();
    } catch (error) {
      console.error('CourseReminderService: Failed to save settings:', error);
    }
  }

  /**
   * 从本地存储加载活跃提醒记录
   */
  private async loadActiveReminders(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const savedReminders = await this.dataPreferences.get('activeReminders', '') as string;
      if (savedReminders) {
        const reminders: ReminderRecord[] = JSON.parse(savedReminders);
        reminders.forEach(r => {
          this.activeReminders.set(r.courseKey, r);
        });
      }
    } catch (error) {
      console.error('CourseReminderService: Failed to load active reminders:', error);
    }
  }

  /**
   * 保存活跃提醒记录到本地存储
   */
  private async saveActiveReminders(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const reminders: ReminderRecord[] = Array.from(this.activeReminders.values());
      await this.dataPreferences.put('activeReminders', JSON.stringify(reminders));
      await this.dataPreferences.flush();
    } catch (error) {
      console.error('CourseReminderService: Failed to save active reminders:', error);
    }
  }

  /**
   * 获取当前提醒设置
   * 
   * @returns 提醒设置的副本
   */
  getSettings(): CourseReminderSettings {
    const result: CourseReminderSettings = {
      enabled: this.settings.enabled,
      advanceMinutes: this.settings.advanceMinutes,
      remindBeforeClass: this.settings.remindBeforeClass,
      remindTomorrowCourse: this.settings.remindTomorrowCourse,
      nightReminderTime: this.settings.nightReminderTime
    };
    return result;
  }

  /**
   * 更新提醒设置
   * 
   * @param newSettings - 新的设置（部分字段）
   */
  async updateSettings(newSettings: Partial<CourseReminderSettings>): Promise<void> {
    if (newSettings.enabled !== undefined) this.settings.enabled = newSettings.enabled;
    if (newSettings.advanceMinutes !== undefined) this.settings.advanceMinutes = newSettings.advanceMinutes;
    if (newSettings.remindBeforeClass !== undefined) this.settings.remindBeforeClass = newSettings.remindBeforeClass;
    if (newSettings.remindTomorrowCourse !== undefined) this.settings.remindTomorrowCourse = newSettings.remindTomorrowCourse;
    if (newSettings.nightReminderTime !== undefined) this.settings.nightReminderTime = newSettings.nightReminderTime;
    await this.saveSettings();
    console.info('CourseReminderService: Settings updated');
  }

  /**
   * 请求通知权限
   * 
   * @returns 是否获得权限
   */
  async requestNotificationPermission(): Promise<boolean> {
    try {
      const granted = await notificationManager.requestEnableNotification();
      console.info('CourseReminderService: Notification permission granted:', granted);
      return true;
    } catch (error) {
      console.error('CourseReminderService: Failed to request notification permission:', error);
      return false;
    }
  }

  /**
   * 检查是否具有通知权限
   * 
   * @returns 是否具有权限
   */
  async hasNotificationPermission(): Promise<boolean> {
    try {
      const enabled = await notificationManager.isNotificationEnabled();
      return enabled;
    } catch (error) {
      console.error('CourseReminderService: Failed to check notification permission:', error);
      return false;
    }
  }

  /**
   * 生成课程唯一标识键
   * 
   * 格式：week_dayOfWeek_startTime_courseName
   * 
   * @param course - 课程信息
   * @returns 唯一标识键
   */
  private generateCourseKey(course: CourseReminderInfo): string {
    return `${course.week}_${course.dayOfWeek}_${course.startTime}_${course.courseName}`;
  }

  /**
   * 将时间字符串转换为分钟数
   * 
   * @param timeStr - 时间字符串（格式：HH:mm）
   * @returns 分钟数（从0点开始计算）
   */
  private parseTimeToMinutes(timeStr: string): number {
    const parts = timeStr.split(':');
    if (parts.length !== 2) return 0;
    const hours = parseInt(parts[0]);
    const minutes = parseInt(parts[1]);
    return hours * 60 + minutes;
  }

  /**
   * 统一计算有效触发时间
   * 
   * 根据是否提供学期开始日期，自动选择绝对时间或相对时间计算策略，
   * 确保过期判断和提醒发布使用同一时间源，避免双重计算不一致
   * 
   * @param course - 课程信息
   * @param advanceMinutes - 提前提醒分钟数
   * @param semesterStartDate - 学期开始日期（可选）
   * @returns 触发时间计算结果
   */
  private calculateEffectiveTriggerTime(course: CourseReminderInfo, advanceMinutes: number, semesterStartDate?: string): TriggerTimeResult {
    let triggerTimeSeconds: number;
    
    if (semesterStartDate) {
      triggerTimeSeconds = this.calculateAbsoluteTriggerTime(course, advanceMinutes, semesterStartDate);
      console.info('CourseReminderService: Using absolute trigger time strategy, triggerTimeSeconds:', triggerTimeSeconds);
    } else {
      const triggerDate = this.calculateTriggerDateTime(course, advanceMinutes);
      triggerTimeSeconds = Math.floor(triggerDate.getTime() / 1000);
      console.info('CourseReminderService: Using relative trigger time strategy, triggerTimeSeconds:', triggerTimeSeconds);
    }
    
    const nowSeconds = Math.floor(Date.now() / 1000);
    const isExpired = triggerTimeSeconds <= nowSeconds;
    const triggerDate = new Date(triggerTimeSeconds * 1000);
    
    return { triggerTimeSeconds, isExpired, triggerDate };
  }

  /**
   * 计算距离提醒时间的秒数
   * 
   * 基于当前时间计算到下次课程提醒还需要多少秒
   * 
   * @param course - 课程信息
   * @param advanceMinutes - 提前提醒分钟数
   * @returns 秒数
   */
  private calculateSecondsToReminder(course: CourseReminderInfo, advanceMinutes: number): number {
    const now = new Date();
    const currentDay = now.getDay();
    const currentDayOfWeek = currentDay === 0 ? 7 : currentDay;

    const currentMinutes = now.getHours() * 60 + now.getMinutes();

    const courseStartMinutes = this.parseTimeToMinutes(course.startTime);
    let reminderMinutes = courseStartMinutes - advanceMinutes;
    if (reminderMinutes < 0) {
      console.warn('CourseReminderService: advanceMinutes exceeds course start time, clamping to 00:00');
      reminderMinutes = 0;
    }

    let daysUntilClass = course.dayOfWeek - currentDayOfWeek;
    if (daysUntilClass < 0) {
      daysUntilClass += 7;
    } else if (daysUntilClass === 0 && reminderMinutes <= currentMinutes) {
      daysUntilClass = 7;
    }

    const targetMinutes = daysUntilClass * 24 * 60 + reminderMinutes;
    const secondsUntilReminder = (targetMinutes - currentMinutes) * 60;

    return Math.max(0, secondsUntilReminder);
  }

  /**
   * 计算提醒触发时间
   * 
   * 基于当前时间和课程信息计算具体的提醒触发时间
   * 
   * @param course - 课程信息
   * @param advanceMinutes - 提前提醒分钟数
   * @returns 触发时间 Date 对象
   */
  private calculateTriggerDateTime(course: CourseReminderInfo, advanceMinutes: number): Date {
    const now = new Date();
    const currentDay = now.getDay();
    const currentDayOfWeek = currentDay === 0 ? 7 : currentDay;

    const courseStartMinutes = this.parseTimeToMinutes(course.startTime);
    let reminderMinutes = courseStartMinutes - advanceMinutes;
    if (reminderMinutes < 0) {
      console.warn('CourseReminderService: advanceMinutes exceeds course start time, clamping to 00:00');
      reminderMinutes = 0;
    }
    const reminderHour = Math.floor(reminderMinutes / 60);
    const reminderMinute = reminderMinutes % 60;

    let daysUntilClass = course.dayOfWeek - currentDayOfWeek;
    if (daysUntilClass < 0) {
      daysUntilClass += 7;
    } else if (daysUntilClass === 0 && reminderMinutes <= (now.getHours() * 60 + now.getMinutes())) {
      daysUntilClass = 7;
    }

    const triggerDate = new Date(now);
    triggerDate.setDate(triggerDate.getDate() + daysUntilClass);
    triggerDate.setHours(reminderHour, reminderMinute, 0, 0);

    return triggerDate;
  }

  /**
   * 获取指定周的日期映射
   * 
   * 根据学期开始日期计算指定周每天对应的日期
   * 
   * @param week - 周次
   * @param semesterStartDate - 学期开始日期（格式：YYYY-MM-DD）
   * @returns Map<dayOfWeek, Date> 映射表
   */
  private getWeekDates(week: number, semesterStartDate: string): Map<number, Date> {
    const weekDates = new Map<number, Date>();
    
    if (!semesterStartDate) {
      const now = new Date();
      const currentDayOfWeek = now.getDay() || 7;
      for (let day = 1; day <= 7; day++) {
        const date = new Date(now);
        const diff = day - currentDayOfWeek;
        date.setDate(date.getDate() + diff);
        weekDates.set(day, date);
      }
      return weekDates;
    }

    const startDate = new Date(semesterStartDate);
    const weekStartDate = new Date(startDate);
    weekStartDate.setDate(startDate.getDate() + (week - 1) * 7);

    for (let day = 1; day <= 7; day++) {
      const date = new Date(weekStartDate);
      date.setDate(weekStartDate.getDate() + (day - 1));
      weekDates.set(day, date);
    }

    return weekDates;
  }

  /**
   * 计算绝对触发时间
   * 
   * 基于学期开始日期计算提醒的绝对触发时间戳
   * 用于创建日历类型的提醒
   * 
   * @param course - 课程信息
   * @param advanceMinutes - 提前提醒分钟数
   * @param semesterStartDate - 学期开始日期
   * @returns 触发时间戳（秒）
   */
  private calculateAbsoluteTriggerTime(course: CourseReminderInfo, advanceMinutes: number, semesterStartDate: string): number {
    const weekDates = this.getWeekDates(course.week, semesterStartDate);
    const courseDate = weekDates.get(course.dayOfWeek);
    
    if (!courseDate) {
      return 0;
    }

    const courseStartMinutes = this.parseTimeToMinutes(course.startTime);
    let reminderMinutes = courseStartMinutes - advanceMinutes;
    if (reminderMinutes < 0) {
      console.warn('CourseReminderService: advanceMinutes exceeds course start time, clamping to 00:00');
      reminderMinutes = 0;
    }
    const reminderHour = Math.floor(reminderMinutes / 60);
    const reminderMinute = reminderMinutes % 60;

    const triggerDate = new Date(courseDate);
    triggerDate.setHours(reminderHour, reminderMinute, 0, 0);

    return Math.floor(triggerDate.getTime() / 1000);
  }

  /**
   * 发布单个课程提醒
   * 
   * 核心方法，创建一个课程提醒
   * 
   * 处理流程：
   * 1. 检查提醒是否启用
   * 2. 检查通知权限
   * 3. 检查提醒数量限制
   * 4. 处理已存在的同名提醒
   * 5. 计算触发时间
   * 6. 创建提醒（日历类型或计时器类型）
   * 7. 保存提醒记录
   * 
   * @param course - 课程信息
   * @param semesterStartDate - 学期开始日期（可选，提供时使用日历类型提醒）
   * @returns 提醒ID，失败返回 null
   */
  async publishCourseReminder(course: CourseReminderInfo, semesterStartDate?: string): Promise<number | null> {
    console.info('CourseReminderService: publishCourseReminder called for:', course.courseName, 
      'week:', course.week, 'day:', course.dayOfWeek, 'startTime:', course.startTime);
    
    if (!this.settings.enabled) {
      console.info('CourseReminderService: Reminder disabled, skipping');
      return null;
    }

    const hasPermission = await this.hasNotificationPermission();
    console.info('CourseReminderService: hasNotificationPermission:', hasPermission);
    
    if (!hasPermission) {
      const granted = await this.requestNotificationPermission();
      console.info('CourseReminderService: requestNotificationPermission result:', granted);
      if (!granted) {
        console.error('CourseReminderService: No notification permission');
        return null;
      }
    }

    if (this.activeReminders.size >= this.maxRemindersLimit) {
      console.warn('CourseReminderService: Reminder limit reached, cannot set more reminders');
      if (this.context) {
        promptAction.showToast({
          message: `提醒数量已达上限（${this.maxRemindersLimit}个），请清理部分提醒`
        });
      }
      return null;
    }

    const courseKey = this.generateCourseKey(course);
    console.info('CourseReminderService: courseKey:', courseKey);
    
    const existingReminder = this.activeReminders.get(courseKey);
    if (existingReminder) {
      const existingTriggerTime = existingReminder.triggerTime ? new Date(existingReminder.triggerTime * 1000) : null;
      const now = new Date();
      
      if (existingTriggerTime && existingTriggerTime.getTime() <= now.getTime()) {
        console.info('CourseReminderService: Existing reminder already expired, removing from records:', 
          existingReminder.reminderId, 'triggerTime:', existingTriggerTime.toLocaleString());
        this.activeReminders.delete(courseKey);
      } else {
        try {
          await reminderAgentManager.cancelReminder(existingReminder.reminderId);
          console.info('CourseReminderService: Cancelled existing reminder:', existingReminder.reminderId,
            'triggerTime:', existingTriggerTime?.toLocaleString());
        } catch (error) {
          console.warn('CourseReminderService: Failed to cancel existing reminder (may already triggered):', error);
          this.activeReminders.delete(courseKey);
        }
      }
    }

    const now = new Date();
    const triggerResult = this.calculateEffectiveTriggerTime(course, this.settings.advanceMinutes, semesterStartDate);
    
    console.info('CourseReminderService: Effective trigger time:', triggerResult.triggerDate.toLocaleString(),
      'isExpired:', triggerResult.isExpired, 'advanceMinutes:', this.settings.advanceMinutes);
    
    if (triggerResult.isExpired) {
      console.info('CourseReminderService: Reminder time already passed, skipping');
      return null;
    }

    try {
      const homePageWantAgent: ESObject | null = await this.createWantAgentForHomePage();
      const reminderContent = this.buildRichReminderContent(course);

      let reminderId: number;

      if (semesterStartDate) {
        const triggerDate = triggerResult.triggerDate;
        const dateTime: reminderAgentManager.LocalDateTime = {
          year: triggerDate.getFullYear(),
          month: triggerDate.getMonth() + 1,
          day: triggerDate.getDate(),
          hour: triggerDate.getHours(),
          minute: triggerDate.getMinutes(),
          second: 0
        };

        const reminderRequest: reminderAgentManager.ReminderRequestCalendar = {
          reminderType: reminderAgentManager.ReminderType.REMINDER_TYPE_CALENDAR,
          dateTime: dateTime,
          title: course.courseName,
          content: reminderContent,
          expiredContent: `${course.courseName} 课程提醒已过期`,
          notificationId: this.generateNotificationId(course),
          slotType: notificationManager.SlotType.SOCIAL_COMMUNICATION,
          tapDismissed: true
        };

        if (homePageWantAgent) {
          reminderRequest.wantAgent = homePageWantAgent;
        }

        reminderId = await reminderAgentManager.publishReminder(reminderRequest);
        console.info('CourseReminderService: Published calendar reminder, ID:', reminderId, 
          'triggerTime:', triggerResult.triggerDate.toLocaleString());
      } else {
        const secondsToReminder = triggerResult.triggerTimeSeconds - Math.floor(now.getTime() / 1000);
        
        console.info('CourseReminderService: Timer reminder, secondsToReminder:', secondsToReminder);
        
        const reminderRequest: reminderAgentManager.ReminderRequestTimer = {
          reminderType: reminderAgentManager.ReminderType.REMINDER_TYPE_TIMER,
          triggerTimeInSeconds: secondsToReminder,
          title: course.courseName,
          content: reminderContent,
          expiredContent: `${course.courseName} 课程提醒已过期`,
          notificationId: this.generateNotificationId(course),
          slotType: notificationManager.SlotType.SOCIAL_COMMUNICATION,
          tapDismissed: true
        };

        if (homePageWantAgent) {
          reminderRequest.wantAgent = homePageWantAgent;
        }

        reminderId = await reminderAgentManager.publishReminder(reminderRequest);
        console.info('CourseReminderService: Published timer reminder, ID:', reminderId, 
          'triggerTime:', triggerResult.triggerDate.toLocaleString());
      }

      const record: ReminderRecord = {
        reminderId: reminderId,
        courseKey: courseKey,
        createdAt: Date.now(),
        reminderType: semesterStartDate ? 'calendar' : 'timer',
        triggerTime: triggerResult.triggerTimeSeconds,
        courseName: course.courseName,
        location: course.location,
        teacher: course.teacher
      };
      this.activeReminders.set(courseKey, record);
      await this.saveActiveReminders();

      return reminderId;
    } catch (error) {
      console.error('CourseReminderService: Failed to publish reminder:', error);
      
      const businessError = error as BusinessError;
      if (businessError.code === 1700002) {
        console.warn('CourseReminderService: Reminder limit exceeded (error 1700002)');
        if (this.context) {
          promptAction.showToast({ 
            message: `提醒数量已达系统限制（${this.maxRemindersLimit}个）` 
          });
        }
      }
      
      return null;
    }
  }

  /**
   * 构建富文本提醒内容
   * 
   * 格式：
   * ```
   * 08:30 - 10:05
   * 地点：教学楼A-101
   * 教师：张老师
   * 类型：专业课
   * ```
   * 
   * @param course - 课程信息
   * @returns 格式化的提醒内容
   */
  private buildRichReminderContent(course: CourseReminderInfo): string {
    const timeInfo = `${course.startTime} - ${course.endTime}`;
    const locationInfo = course.location ? `地点：${course.location}` : '地点：待定';
    const teacherInfo = course.teacher ? `教师：${course.teacher}` : '教师：待定';
    
    let content = `${timeInfo}\n${locationInfo}\n${teacherInfo}`;
    
    if (course.courseType) {
      content += `\n类型：${course.courseType}`;
    }
    
    return content;
  }

  /**
   * 构建详细提醒内容（带图标）
   * 
   * 使用 emoji 图标美化显示
   * 
   * @param course - 课程信息
   * @returns 格式化的提醒内容
   */
  private buildDetailedReminderContent(course: CourseReminderInfo): string {
    const lines: string[] = [];
    
    lines.push(`🕐 ${course.startTime} - ${course.endTime}`);
    lines.push(`📍 ${course.location || '地点待定'}`);
    lines.push(`👨‍🏫 ${course.teacher || '教师待定'}`);
    
    if (course.courseType) {
      lines.push(`📚 ${course.courseType}`);
    }
    
    return lines.join('\n');
  }

  /**
   * 生成通知ID
   * 
   * 使用课程信息的哈希值生成唯一的通知ID
   * 确保同一课程的提醒使用相同的通知ID
   * 
   * @param course - 课程信息
   * @returns 通知ID（0-999999）
   */
  private generateNotificationId(course: CourseReminderInfo): number {
    const courseInfo = `${course.courseName}_${course.week}_${course.dayOfWeek}_${course.startTime}`;
    let hash = 0;
    for (let i = 0; i < courseInfo.length; i++) {
      const char = courseInfo.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash) % 1000000;
  }

  /**
   * 批量发布周课程提醒
   * 
   * 为一周内的所有课程设置提醒
   * 会先清理过期提醒，然后逐个创建新提醒
   * 
   * @param courses - 课程列表
   * @param nextWeekCourses - 下一周课程列表（可选，用于跨周明日提醒）
   * @param semesterStartDate - 学期开始日期（可选）
   */
  async publishWeeklyReminders(courses: CourseReminderInfo[], nextWeekCourses?: CourseReminderInfo[], semesterStartDate?: string): Promise<void> {
    if (!this.settings.enabled) {
      console.info('CourseReminderService: Reminder disabled, skipping batch publish');
      return;
    }

    await this.cleanupExpiredReminders();

    // 取消所有非当前周的旧提醒，避免跨周后旧提醒残留
    const currentWeek = this.getCurrentWeek();
    const oldWeekKeys: string[] = [];
    this.activeReminders.forEach((record: ReminderRecord, key: string) => {
      const recordWeek = this.extractWeekFromCourseKey(key);
      if (recordWeek > 0 && recordWeek !== currentWeek) {
        oldWeekKeys.push(key);
      }
    });
    
    for (const key of oldWeekKeys) {
      const record = this.activeReminders.get(key);
      if (record) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          console.info('CourseReminderService: Cancelled old week reminder before publish:', key);
        } catch (error) {
          console.warn('CourseReminderService: Failed to cancel old week reminder:', key, error);
        }
        this.activeReminders.delete(key);
      }
    }
    
    if (oldWeekKeys.length > 0) {
      await this.saveActiveReminders();
      console.info('CourseReminderService: Cancelled', oldWeekKeys.length, 'old week reminders before publishing new week');
    }

    let successCount = 0;
    let failCount = 0;
    let limitReached = false;

    if (this.settings.remindBeforeClass) {
      for (const course of courses) {
        if (limitReached) {
          console.info('CourseReminderService: Reminder limit reached, skipping remaining courses');
          break;
        }

        const result = await this.publishCourseReminder(course, semesterStartDate);
        if (result !== null) {
          successCount++;
        } else {
          failCount++;
          
          if (this.activeReminders.size >= this.maxRemindersLimit) {
            limitReached = true;
            console.warn('CourseReminderService: Reminder limit reached during batch publish');
          }
        }
      }
    }

    if (this.settings.remindTomorrowCourse && !limitReached) {
      const tomorrowResult = await this.publishTomorrowCourseReminders(courses, nextWeekCourses);
      if (tomorrowResult > 0) {
        successCount += tomorrowResult;
      }
    }

    console.info(`CourseReminderService: Published ${successCount} reminders, failed ${failCount}`);
    
    if (limitReached && this.context) {
      promptAction.showToast({ 
        message: `已设置 ${successCount} 个提醒（系统限制：${this.maxRemindersLimit}个）` 
      });
    }
  }

  /**
   * 生成早晨提醒通知ID
   * 
   * @param dayOfWeek - 星期几
   * @param week - 周次
   * @returns 通知ID
   */
  private generateMorningNotificationId(dayOfWeek: number, week: number): number {
    return 3000000 + week * 10 + dayOfWeek;
  }

  /**
   * 获取即将触发的提醒列表
   * 
   * 返回所有未触发的提醒，按触发时间排序
   * 
   * @returns 即将触发的提醒数组
   */
  getUpcomingReminders(): UpcomingReminder[] {
    const upcoming: UpcomingReminder[] = [];
    const now = new Date();

    this.activeReminders.forEach((record: ReminderRecord, key: string) => {
      if (record.triggerTime) {
        const triggerDate = new Date(record.triggerTime * 1000);
        
        if (triggerDate > now) {
          let type: 'before_class' | 'morning' | 'tomorrow' = 'before_class';
          if (key.startsWith('morning_')) {
            type = 'morning';
          } else if (key.startsWith('tomorrow_')) {
            type = 'tomorrow';
          }

          const parts = key.split('_');
          upcoming.push({
            courseName: parts.length > 2 ? parts.slice(2).join('_') : key,
            location: '',
            teacher: '',
            triggerTime: triggerDate,
            type: type
          });
        }
      }
    });

    return upcoming.sort((a, b) => a.triggerTime.getTime() - b.triggerTime.getTime());
  }

  /**
   * 获取提醒统计摘要
   * 
   * @returns 包含各类提醒数量的统计信息
   */
  getRemindersSummary(): RemindersSummary {
    let beforeClass = 0;
    let morning = 0;
    let tomorrow = 0;

    this.activeReminders.forEach((_: ReminderRecord, key: string) => {
      if (key.startsWith('morning_')) {
        morning++;
      } else if (key.startsWith('tomorrow_')) {
        tomorrow++;
      } else {
        beforeClass++;
      }
    });

    const summary: RemindersSummary = {
      total: this.activeReminders.size,
      beforeClass: beforeClass,
      morning: morning,
      tomorrow: tomorrow,
      remaining: this.maxRemindersLimit - this.activeReminders.size
    };
    return summary;
  }

  /**
   * 发布明日课程提醒
   * 
   * 在夜间指定时间发送提醒，告知明天的课程安排
   * 
   * @param courses - 课程列表
   * @param nextWeekCourses - 下一周课程列表（可选，用于跨周明日提醒）
   * @param semesterStartDate - 学期开始日期（可选）
   * @returns 成功创建的提醒数量
   */
  private async publishTomorrowCourseReminders(courses: CourseReminderInfo[], nextWeekCourses?: CourseReminderInfo[], semesterStartDate?: string): Promise<number> {
    if (!this.settings.remindTomorrowCourse) {
      return 0;
    }

    const now = new Date();
    const currentDay = now.getDay();
    const currentDayOfWeek = currentDay === 0 ? 7 : currentDay;
    const currentMinutes = now.getHours() * 60 + now.getMinutes();

    // 计算明天的星期几
    const tomorrowDayOfWeek = currentDayOfWeek === 7 ? 1 : currentDayOfWeek + 1;
    // 判断明天是否属于下一周（今天是周日时，明天是下周一）
    const isTomorrowNextWeek = currentDayOfWeek === 7;

    console.info('CourseReminderService: Tomorrow reminder - currentDayOfWeek:', currentDayOfWeek,
      'tomorrowDayOfWeek:', tomorrowDayOfWeek, 'isTomorrowNextWeek:', isTomorrowNextWeek);

    // 选择数据源：跨周时使用下一周数据
    const effectiveCourses = isTomorrowNextWeek ? (nextWeekCourses || []) : courses;
    const tomorrowCourses = effectiveCourses.filter(c => c.dayOfWeek === tomorrowDayOfWeek);

    if (tomorrowCourses.length === 0) {
      console.info('CourseReminderService: No courses tomorrow, skipping tomorrow reminder');
      return 0;
    }

    const timeParts = this.settings.nightReminderTime.split(':');
    const nightHour = parseInt(timeParts[0]);
    const nightMinute = parseInt(timeParts[1]);
    const nightMinutes = nightHour * 60 + nightMinute;

    // 计算到今晚夜间提醒时间的秒数
    // daysUntilNight = 0 表示今晚（今天夜间提醒明天课程）
    let daysUntilNight = 0;
    const targetMinutes = daysUntilNight * 24 * 60 + nightMinutes;
    const secondsUntilReminder = (targetMinutes - currentMinutes) * 60;

    if (secondsUntilReminder <= 0) {
      console.info('CourseReminderService: Tonight reminder time already passed, skipping tomorrow reminder');
      return 0;
    }

    const nightKey = `tomorrow_${tomorrowDayOfWeek}`;
    const existingNightReminder = this.activeReminders.get(nightKey);
    if (existingNightReminder) {
      try {
        await reminderAgentManager.cancelReminder(existingNightReminder.reminderId);
        this.activeReminders.delete(nightKey);
        console.info('CourseReminderService: Cancelled existing tomorrow course reminder for update');
      } catch (error) {
        console.warn('CourseReminderService: Failed to cancel existing tomorrow course reminder:', error);
      }
    }

    if (this.activeReminders.size >= this.maxRemindersLimit) {
      console.warn('CourseReminderService: Tomorrow course reminder limit reached');
      return 0;
    }

    try {
      const homePageWantAgent: ESObject | null = await this.createWantAgentForHomePage();
      
      const courseInfo = tomorrowCourses.map(c => `${c.courseName} (${c.startTime}-${c.endTime})`).join('\n');
      const courseCount = tomorrowCourses.length;
      
      const dayNames = ['', '周一', '周二', '周三', '周四', '周五', '周六', '周日'];
      const reminderContent = `明天${dayNames[tomorrowDayOfWeek]}有 ${courseCount} 门课程\n${courseInfo}\n点击查看详情`;

      const reminderRequest: reminderAgentManager.ReminderRequestTimer = {
        reminderType: reminderAgentManager.ReminderType.REMINDER_TYPE_TIMER,
        triggerTimeInSeconds: secondsUntilReminder,
        title: '明日课程提醒',
        content: reminderContent,
        expiredContent: '明日课程提醒已过期',
        notificationId: this.generateTomorrowNotificationId(tomorrowDayOfWeek),
        slotType: notificationManager.SlotType.SOCIAL_COMMUNICATION,
        wantAgent: homePageWantAgent || undefined,
        tapDismissed: true
      };

      const reminderId = await reminderAgentManager.publishReminder(reminderRequest);
      console.info('CourseReminderService: Published tomorrow course reminder, ID:', reminderId, 'for day:', tomorrowDayOfWeek);

      const record: ReminderRecord = {
        reminderId: reminderId,
        courseKey: nightKey,
        createdAt: Date.now(),
        reminderType: 'timer',
        triggerTime: Math.floor(now.getTime() / 1000) + secondsUntilReminder
      };
      this.activeReminders.set(nightKey, record);
      await this.saveActiveReminders();

      return 1;
    } catch (error) {
      console.error('CourseReminderService: Failed to publish tomorrow course reminder:', error);
      return 0;
    }
  }

  /**
   * 生成明日提醒通知ID
   * 
   * @param dayOfWeek - 星期几
   * @returns 通知ID
   */
  private generateTomorrowNotificationId(dayOfWeek: number): number {
    return 2000000 + dayOfWeek;
  }

  /**
   * 取消单个课程提醒
   * 
   * @param course - 课程信息
   * @returns 是否成功取消
   */
  async cancelCourseReminder(course: CourseReminderInfo): Promise<boolean> {
    const courseKey = this.generateCourseKey(course);
    const record = this.activeReminders.get(courseKey);

    if (!record) {
      console.info('CourseReminderService: No active reminder found for course');
      return false;
    }

    try {
      await reminderAgentManager.cancelReminder(record.reminderId);
      this.activeReminders.delete(courseKey);
      await this.saveActiveReminders();
      console.info('CourseReminderService: Cancelled reminder for course');
      return true;
    } catch (error) {
      console.error('CourseReminderService: Failed to cancel reminder:', error);
      return false;
    }
  }

  /**
   * 取消所有明日课程提醒
   * 
   * @returns 取消的提醒数量
   */
  async cancelTomorrowReminders(): Promise<number> {
    let cancelledCount = 0;
    
    const entries = Array.from(this.activeReminders.entries());
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const key = entry[0];
      const record = entry[1];
      
      if (key.startsWith('tomorrow_')) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          this.activeReminders.delete(key);
          cancelledCount++;
          console.info('CourseReminderService: Cancelled tomorrow course reminder:', record.reminderId);
        } catch (error) {
          console.error('CourseReminderService: Failed to cancel tomorrow course reminder:', error);
        }
      }
    }
    
    if (cancelledCount > 0) {
      await this.saveActiveReminders();
    }
    
    return cancelledCount;
  }

  /**
   * 取消所有早晨提醒
   * 
   * @returns 取消的提醒数量
   */
  async cancelMorningReminders(): Promise<number> {
    let cancelledCount = 0;
    
    const entries = Array.from(this.activeReminders.entries());
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const key = entry[0];
      const record = entry[1];
      
      if (key.startsWith('morning_')) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          this.activeReminders.delete(key);
          cancelledCount++;
          console.info('CourseReminderService: Cancelled morning course reminder:', record.reminderId);
        } catch (error) {
          console.error('CourseReminderService: Failed to cancel morning course reminder:', error);
        }
      }
    }
    
    if (cancelledCount > 0) {
      await this.saveActiveReminders();
    }
    
    return cancelledCount;
  }

  /**
   * 取消所有课前提醒
   * 
   * 课前提醒是指非早晨提醒、非明日提醒的课程提醒
   * 
   * @returns 取消的提醒数量
   */
  async cancelBeforeClassReminders(): Promise<number> {
    let cancelledCount = 0;
    
    const entries = Array.from(this.activeReminders.entries());
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const key = entry[0];
      const record = entry[1];
      
      if (!key.startsWith('morning_') && !key.startsWith('tomorrow_')) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          this.activeReminders.delete(key);
          cancelledCount++;
          console.info('CourseReminderService: Cancelled before class reminder:', record.reminderId);
        } catch (error) {
          console.error('CourseReminderService: Failed to cancel before class reminder:', error);
        }
      }
    }
    
    if (cancelledCount > 0) {
      await this.saveActiveReminders();
    }
    
    return cancelledCount;
  }

  /**
   * 取消所有提醒
   * 
   * 清除所有活跃提醒，包括课前提醒、早晨提醒和明日提醒
   */
  async cancelAllReminders(): Promise<void> {
    try {
      for (const record of this.activeReminders.values()) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
        } catch (error) {
          console.warn('CourseReminderService: Failed to cancel reminder:', record.reminderId, error);
        }
      }
      this.activeReminders.clear();
      await this.saveActiveReminders();
      console.info('CourseReminderService: Cancelled all reminders');
      
      if (this.context) {
        promptAction.showToast({ message: '已清除所有提醒' });
      }
    } catch (error) {
      console.error('CourseReminderService: Failed to cancel all reminders:', error);
      if (this.context) {
        promptAction.showToast({ message: '清除提醒失败' });
      }
    }
  }

  /**
   * 获取活跃提醒数量
   * 
   * @returns 活跃提醒数量
   */
  getActiveRemindersCount(): number {
    return this.activeReminders.size;
  }

  /**
   * 获取提醒数量上限
   * 
   * @returns 提醒数量上限
   */
  getMaxRemindersLimit(): number {
    return this.maxRemindersLimit;
  }

  /**
   * 获取剩余提醒容量
   * 
   * @returns 剩余可创建的提醒数量
   */
  getRemainingReminderCapacity(): number {
    return Math.max(0, this.maxRemindersLimit - this.activeReminders.size);
  }

  /**
   * 检查课程是否有活跃提醒
   * 
   * @param course - 课程信息
   * @returns 是否存在活跃提醒
   */
  hasActiveReminder(course: CourseReminderInfo): boolean {
    const courseKey = this.generateCourseKey(course);
    return this.activeReminders.has(courseKey);
  }

  /**
   * 刷新指定周的提醒
   * 
   * @param courses - 课程列表
   * @param week - 周次
   */
  async refreshRemindersForWeek(courses: CourseReminderInfo[], week: number): Promise<void> {
    const weekCourses = courses.filter(c => c.week === week);
    
    if (weekCourses.length > this.getRemainingReminderCapacity()) {
      console.warn('CourseReminderService: Not enough capacity for week reminders');
      if (this.context) {
        promptAction.showToast({ 
          message: `本周有 ${weekCourses.length} 个课程，但只能设置 ${this.getRemainingReminderCapacity()} 个提醒` 
        });
      }
    }
    
    await this.publishWeeklyReminders(weekCourses);
  }

  /**
   * 应用前台时刷新提醒
   * 
   * 当应用从后台切换到前台时调用
   * 会清理过期提醒并触发提醒刷新
   * 检测到周次变化时，自动取消旧周提醒并重建当前周提醒
   */
  async refreshRemindersOnForeground(): Promise<void> {
    if (!this.settings.enabled) {
      console.info('CourseReminderService: Reminder disabled, skip auto refresh');
      return;
    }

    const now = Date.now();
    const currentWeek = this.getCurrentWeek();
    
    // 检测周次变化：跨周时需要取消旧提醒并重建
    if (this.refreshState.lastRefreshWeek > 0 && this.refreshState.lastRefreshWeek !== currentWeek) {
      console.info('CourseReminderService: Week changed from', this.refreshState.lastRefreshWeek, 
        'to', currentWeek, ', cancelling old week reminders');
      await this.cancelRemindersForWeek(this.refreshState.lastRefreshWeek);
    }

    if (now - this.refreshState.lastRefreshTime < this.MIN_REFRESH_INTERVAL &&
        this.refreshState.lastRefreshWeek === currentWeek) {
      console.info('CourseReminderService: Skip refresh, too frequent');
      return;
    }

    console.info('CourseReminderService: Auto refreshing reminders on foreground');
    
    this.refreshState.lastRefreshTime = now;
    this.refreshState.lastRefreshWeek = currentWeek;
    
    await this.syncActiveRemindersWithSystem();
    await this.cleanupExpiredReminders();
    
    AppStorage.SetOrCreate('needRefreshReminders', true);
  }

  /**
   * 获取当前周次
   * 
   * 从 AppStorage 中读取学期设置计算当前周次
   * 
   * @returns 当前周次，默认返回 1
   */
  private getCurrentWeek(): number {
    const settings = AppStorage.Get('scheduleSettings') as SettingsData | undefined;
    if (settings && typeof settings.getCurrentWeek === 'function') {
      return settings.getCurrentWeek();
    }
    return 1;
  }

  /**
   * 从courseKey中提取周次
   * 
   * courseKey格式：week_dayOfWeek_startTime_courseName
   * 或：tomorrow_dayOfWeek / morning_week_dayOfWeek
   * 
   * @param courseKey - 课程唯一标识键
   * @returns 周次，无法提取时返回 -1
   */
  private extractWeekFromCourseKey(courseKey: string): number {
    if (courseKey.startsWith('tomorrow_') || courseKey.startsWith('morning_')) {
      const parts = courseKey.split('_');
      if (parts.length >= 2 && courseKey.startsWith('morning_')) {
        const week = parseInt(parts[1]);
        if (!isNaN(week)) return week;
      }
      return -1;
    }
    const parts = courseKey.split('_');
    if (parts.length >= 1) {
      const week = parseInt(parts[0]);
      if (!isNaN(week)) return week;
    }
    return -1;
  }

  /**
   * 取消指定周的所有提醒
   * 
   * 用于跨周时清理旧周的提醒，避免过期通知残留
   * 
   * @param week - 要取消的周次
   */
  private async cancelRemindersForWeek(week: number): Promise<void> {
    const keysToRemove: string[] = [];
    
    this.activeReminders.forEach((record: ReminderRecord, key: string) => {
      const recordWeek = this.extractWeekFromCourseKey(key);
      if (recordWeek === week) {
        keysToRemove.push(key);
      }
    });
    
    if (keysToRemove.length === 0) {
      console.info('CourseReminderService: No reminders to cancel for week', week);
      return;
    }
    
    console.info('CourseReminderService: Cancelling', keysToRemove.length, 'reminders for week', week);
    
    for (const key of keysToRemove) {
      const record = this.activeReminders.get(key);
      if (record) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          console.info('CourseReminderService: Cancelled reminder for week', week, ':', key);
        } catch (error) {
          console.warn('CourseReminderService: Failed to cancel reminder:', key, 'error:', error);
        }
        this.activeReminders.delete(key);
      }
    }
    
    await this.saveActiveReminders();
    console.info('CourseReminderService: Cancelled all reminders for week', week);
  }

  /**
   * 清理过期提醒
   * 
   * 内部方法，清理以下类型的提醒：
   * 1. 触发时间已过的提醒
   * 2. 创建时间超过 7 天的提醒
   */
  private async cleanupExpiredReminders(): Promise<void> {
    const now = Date.now();
    const nowSeconds = Math.floor(now / 1000);
    const currentWeek = this.getCurrentWeek();
    const expiredKeys: string[] = [];
    
    this.activeReminders.forEach((record: ReminderRecord, key: string) => {
      // 条件1：触发时间已过
      if (record.triggerTime && record.triggerTime <= nowSeconds) {
        expiredKeys.push(key);
        console.info('CourseReminderService: Found expired reminder (trigger time passed):', key, 
          'triggerTime:', new Date(record.triggerTime * 1000).toLocaleString());
      }
      // 条件2：创建时间超过2天（代理提醒最多覆盖一周，2天足够覆盖跨天场景）
      else if (now - record.createdAt > 2 * 24 * 60 * 60 * 1000) {
        expiredKeys.push(key);
        console.info('CourseReminderService: Found old reminder (created > 2 days):', key);
      }
      // 条件3：跨周过期 - courseKey中的周次与当前周不同且小于当前周
      else {
        const recordWeek = this.extractWeekFromCourseKey(key);
        if (recordWeek > 0 && recordWeek < currentWeek) {
          expiredKeys.push(key);
          console.info('CourseReminderService: Found cross-week expired reminder (week', recordWeek, 
            '< current week', currentWeek, '):', key);
        }
      }
    });
    
    for (let i = 0; i < expiredKeys.length; i++) {
      const key = expiredKeys[i];
      const record = this.activeReminders.get(key);
      if (record) {
        try {
          await reminderAgentManager.cancelReminder(record.reminderId);
          console.info('CourseReminderService: Cleaned up expired reminder:', key, 'ID:', record.reminderId);
        } catch (error) {
          console.warn('CourseReminderService: Failed to cancel expired reminder (may already be triggered):', 
            key, 'ID:', record.reminderId, 'error:', error);
        }
        this.activeReminders.delete(key);
      }
    }
    
    if (expiredKeys.length > 0) {
      await this.saveActiveReminders();
      console.info('CourseReminderService: Cleaned up', expiredKeys.length, 'expired reminders');
    }
  }

  /**
   * 清理所有过期提醒（公共方法）
   * 
   * 供外部调用的过期提醒清理方法
   * 
   * @returns 清理后剩余的提醒数量
   */
  async cleanupAllExpiredReminders(): Promise<number> {
    await this.cleanupExpiredReminders();
    return this.activeReminders.size;
  }
}

export { CourseReminderService };
