/**
 * 推送通知服务模块
 * 
 * 提供推送通知管理功能，包括：
 * - 推送Token管理
 * - 推送设置管理
 * - 消息接收处理
 * - 通知分类控制
 * 
 * @module PushNotificationService
 */

import { pushService, pushCommon } from '@kit.PushKit';
import { notificationManager } from '@kit.NotificationKit';
import { preferences } from '@kit.ArkData';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { Ability } from '@kit.AbilityKit';

/** 日志域 */
const DOMAIN = 0x0000;
/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'push_notification_settings';

/**
 * 推送消息接口
 */
export interface PushMessage {
  /** 消息类型 */
  type: string;
  /** 消息标题 */
  title: string;
  /** 消息内容 */
  content: string;
  /** 消息数据 */
  data: string;
  /** 时间戳 */
  timestamp: number;
  /** 跳转URL */
  url?: string;
  /** 关键词 */
  keyword?: string;
}

/**
 * 推送通知设置接口
 */
export interface PushNotificationSettings {
  /** 课程变更通知开关 */
  courseChangeEnabled: boolean;
  /** 新成绩通知开关 */
  newGradeEnabled: boolean;
  /** 新考试通知开关 */
  newExamEnabled: boolean;
  /** 考试提醒开关 */
  examReminderEnabled: boolean;
  /** 公告通知开关 */
  announcementEnabled: boolean;
  /** 推送是否启用 */
  isPushEnabled: boolean;
}

/**
 * 推送通知服务类
 * 
 * 管理推送通知的设置和消息处理
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const pushService = PushNotificationService.getInstance();
 * await pushService.initialize(context);
 * const token = await pushService.getPushToken();
 * ```
 */
class PushNotificationService {
  /** 单例实例 */
  private static instance: PushNotificationService;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 推送Token */
  private pushToken: string = '';
  /** 课程变更通知开关 */
  private courseChangeEnabled: boolean = true;
  /** 新成绩通知开关 */
  private newGradeEnabled: boolean = true;
  /** 新考试通知开关 */
  private newExamEnabled: boolean = true;
  /** 考试提醒开关 */
  private examReminderEnabled: boolean = true;
  /** 公告通知开关 */
  private announcementEnabled: boolean = true;
  /** 推送是否启用 */
  private isPushEnabled: boolean = false;
  /** 应用上下文 */
  private context: Context | null = null;

  private constructor() {}

  static getInstance(): PushNotificationService {
    if (!PushNotificationService.instance) {
      PushNotificationService.instance = new PushNotificationService();
    }
    return PushNotificationService.instance;
  }

  async initialize(context: Context): Promise<void> {
    this.context = context;
    try {
      this.dataPreferences = await preferences.getPreferences(context, PREFERENCES_NAME);
      await this.loadSettings();
      console.info('PushNotificationService: Initialized successfully');
    } catch (error) {
      console.error('PushNotificationService: Failed to initialize:', error);
    }
  }

  private async loadSettings(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const savedSettings = await this.dataPreferences.get('settings', '') as string;
      if (savedSettings) {
        const parsed = JSON.parse(savedSettings) as PushNotificationSettings;
        if (parsed.courseChangeEnabled !== undefined) {
          this.courseChangeEnabled = parsed.courseChangeEnabled;
        }
        if (parsed.newGradeEnabled !== undefined) {
          this.newGradeEnabled = parsed.newGradeEnabled;
        }
        if (parsed.newExamEnabled !== undefined) {
          this.newExamEnabled = parsed.newExamEnabled;
        }
        if (parsed.examReminderEnabled !== undefined) {
          this.examReminderEnabled = parsed.examReminderEnabled;
        }
        if (parsed.announcementEnabled !== undefined) {
          this.announcementEnabled = parsed.announcementEnabled;
        }
        if (parsed.isPushEnabled !== undefined) {
          this.isPushEnabled = parsed.isPushEnabled;
        }
      }
    } catch (error) {
      console.error('PushNotificationService: Failed to load settings:', error);
    }
  }

  private async saveSettings(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const settings: PushNotificationSettings = {
        courseChangeEnabled: this.courseChangeEnabled,
        newGradeEnabled: this.newGradeEnabled,
        newExamEnabled: this.newExamEnabled,
        examReminderEnabled: this.examReminderEnabled,
        announcementEnabled: this.announcementEnabled,
        isPushEnabled: this.isPushEnabled
      };
      await this.dataPreferences.put('settings', JSON.stringify(settings));
      await this.dataPreferences.flush();
    } catch (error) {
      console.error('PushNotificationService: Failed to save settings:', error);
    }
  }

  getSettings(): PushNotificationSettings {
    return {
      courseChangeEnabled: this.courseChangeEnabled,
      newGradeEnabled: this.newGradeEnabled,
      newExamEnabled: this.newExamEnabled,
      examReminderEnabled: this.examReminderEnabled,
      announcementEnabled: this.announcementEnabled,
      isPushEnabled: this.isPushEnabled
    };
  }

  async updateSettings(settings: PushNotificationSettings): Promise<void> {
    this.courseChangeEnabled = settings.courseChangeEnabled;
    this.newGradeEnabled = settings.newGradeEnabled;
    this.newExamEnabled = settings.newExamEnabled;
    this.examReminderEnabled = settings.examReminderEnabled;
    if (settings.announcementEnabled !== undefined) {
      this.announcementEnabled = settings.announcementEnabled;
    }
    if (settings.isPushEnabled !== undefined) {
      this.isPushEnabled = settings.isPushEnabled;
    }
    await this.saveSettings();
    console.info('PushNotificationService: Settings updated');
  }

  async getPushToken(): Promise<string> {
    try {
      if (this.pushToken) {
        return this.pushToken;
      }

      const token = await pushService.getToken();
      this.pushToken = token;
      hilog.info(DOMAIN, 'PushNotificationService', 'Push token obtained: %{public}s', token.substring(0, 20) + '...');
      
      await this.savePushToken(token);
      
      return token;
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to get push token: %{public}s', JSON.stringify(error));
      return '';
    }
  }

  private async savePushToken(token: string): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      await this.dataPreferences.put('pushToken', token);
      await this.dataPreferences.flush();
    } catch (error) {
      console.error('PushNotificationService: Failed to save push token:', error);
    }
  }

  async bindUserAccount(studentId: string): Promise<boolean> {
    try {
      const anonymousId = this.generateAnonymousId(studentId);
      await pushService.bindAppProfileId(2, anonymousId);
      hilog.info(DOMAIN, 'PushNotificationService', 'User account bound successfully');
      return true;
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to bind user account: %{public}s', JSON.stringify(error));
      return false;
    }
  }

  async unbindUserAccount(studentId: string): Promise<boolean> {
    try {
      const anonymousId = this.generateAnonymousId(studentId);
      await pushService.unbindAppProfileId(anonymousId);
      hilog.info(DOMAIN, 'PushNotificationService', 'User account unbound successfully');
      return true;
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to unbind user account: %{public}s', JSON.stringify(error));
      return false;
    }
  }

  private generateAnonymousId(studentId: string): string {
    let hash = 0;
    for (let i = 0; i < studentId.length; i++) {
      const char = studentId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return 'nnlg_' + Math.abs(hash).toString(16);
  }

  registerMessageReceiver(ability: Ability): void {
    try {
      pushService.receiveMessage('DEFAULT', ability, (payload: pushCommon.PushPayload) => {
        this.handlePushMessage(payload);
      });
      hilog.info(DOMAIN, 'PushNotificationService', 'Message receiver registered');
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to register message receiver: %{public}s', JSON.stringify(error));
    }
  }

  private handlePushMessage(payload: pushCommon.PushPayload): void {
    try {
      const jsonData: string = payload.data;
      let messageType: string = 'course_change';
      let title: string = '通知';
      let content: string = '';
      let data: string = '';
      let url: string = '';
      let keyword: string = '';

      if (jsonData) {
        try {
          const parsed: Record<string, string> = JSON.parse(jsonData) as Record<string, string>;
          if (parsed['type']) {
            messageType = parsed['type'];
          }
          if (parsed['title']) {
            title = parsed['title'];
          }
          if (parsed['content']) {
            content = parsed['content'];
          }
          if (parsed['url']) {
            url = parsed['url'];
          }
          if (parsed['keyword']) {
            keyword = parsed['keyword'];
          }
          data = jsonData;
        } catch (e) {
          content = jsonData;
        }
      }

      const message: PushMessage = {
        type: messageType,
        title: title,
        content: content,
        data: data,
        timestamp: Date.now(),
        url: url,
        keyword: keyword
      };

      hilog.info(DOMAIN, 'PushNotificationService', 'Received push message: type=%{public}s, title=%{public}s', 
        message.type, message.title);

      if (!this.isMessageTypeEnabled(message.type)) {
        hilog.info(DOMAIN, 'PushNotificationService', 'Message type %{public}s is disabled, skipping', message.type);
        return;
      }

      this.showLocalNotification(message);
      this.notifyApp(message);
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to handle push message: %{public}s', JSON.stringify(error));
    }
  }

  private isMessageTypeEnabled(type: string): boolean {
    if (type === 'course_change') {
      return this.courseChangeEnabled;
    }
    if (type === 'new_grade') {
      return this.newGradeEnabled;
    }
    if (type === 'new_exam') {
      return this.newExamEnabled;
    }
    if (type === 'exam_reminder') {
      return this.examReminderEnabled;
    }
    if (type === 'announcement') {
      return this.announcementEnabled;
    }
    return true;
  }

  private async showLocalNotification(message: PushMessage): Promise<void> {
    try {
      const notificationRequest: notificationManager.NotificationRequest = {
        id: Date.now() % 1000000,
        content: {
          notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
          normal: {
            title: message.title,
            text: message.content
          }
        },
        deliveryTime: new Date().getTime()
      };

      await notificationManager.publish(notificationRequest);
      hilog.info(DOMAIN, 'PushNotificationService', 'Local notification shown: %{public}s', message.title);
    } catch (error) {
      hilog.error(DOMAIN, 'PushNotificationService', 'Failed to show local notification: %{public}s', JSON.stringify(error));
    }
  }

  private notifyApp(message: PushMessage): void {
    AppStorage.SetOrCreate('pushMessage', JSON.stringify(message));
    AppStorage.SetOrCreate('hasNewPushMessage', true);
    
    if (message.type === 'announcement' && message.url) {
      AppStorage.SetOrCreate('announcementPushUrl', message.url);
      AppStorage.SetOrCreate('announcementPushTitle', message.title);
      hilog.info(DOMAIN, 'PushNotificationService', 'Announcement push stored for navigation: %{public}s', message.url);
    }
  }

  getPushTokenForServer(): string {
    return this.pushToken;
  }

  isPushServiceEnabled(): boolean {
    return this.isPushEnabled;
  }

  async setPushEnabled(enabled: boolean): Promise<void> {
    this.isPushEnabled = enabled;
    await this.saveSettings();
  }
}

export { PushNotificationService };
