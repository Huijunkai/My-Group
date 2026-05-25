/**
 * 通知管理模块
 * 
 * 提供通知状态管理功能，包括：
 * - 新成绩通知状态
 * - 新学分通知状态
 * - 新考试通知状态
 * - 状态持久化
 * 
 * @module NotificationManager
 */

import { preferences } from '@kit.ArkData';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'notification_status';

/**
 * 通知状态接口
 */
interface NotificationStatus {
  /** 是否有新成绩 */
  hasNewGrade: boolean;
  /** 是否有新学分 */
  hasNewCredit: boolean;
  /** 是否有新考试 */
  hasNewExam: boolean;
  /** 最后成绩更新时间 */
  lastGradeUpdate: string;
  /** 最后学分更新时间 */
  lastCreditUpdate: string;
  /** 最后考试更新时间 */
  lastExamUpdate: string;
}

/**
 * 通知管理类
 * 
 * 管理应用内通知状态
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const manager = NotificationManager.getInstance();
 * await manager.initialize(context);
 * manager.setNewGrade(true);
 * ```
 */
class NotificationManager {
  /** 单例实例 */
  private static instance: NotificationManager;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 通知状态 */
  private status: NotificationStatus = {
    hasNewGrade: false,
    hasNewCredit: false,
    hasNewExam: false,
    lastGradeUpdate: '',
    lastCreditUpdate: '',
    lastExamUpdate: ''
  };

  static getInstance(): NotificationManager {
    if (!NotificationManager.instance) {
      NotificationManager.instance = new NotificationManager();
    }
    return NotificationManager.instance;
  }

  async initialize(context: Context): Promise<void> {
    try {
      this.dataPreferences = await preferences.getPreferences(context, PREFERENCES_NAME);
      await this.loadStatus();
      AppStorage.SetOrCreate('notificationStatus', this.status);
    } catch (error) {
      console.error('NotificationManager: Failed to initialize:', error);
    }
  }

  private async loadStatus(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const savedStatus = await this.dataPreferences.get('notificationStatus', '') as string;
      if (savedStatus) {
        this.status = JSON.parse(savedStatus) as NotificationStatus;
      }
    } catch (error) {
      console.error('NotificationManager: Failed to load status:', error);
    }
  }

  private async saveStatus(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      await this.dataPreferences.put('notificationStatus', JSON.stringify(this.status));
      await this.dataPreferences.flush();
      AppStorage.SetOrCreate('notificationStatus', this.status);
    } catch (error) {
      console.error('NotificationManager: Failed to save status:', error);
    }
  }

  setNewGrade(hasNew: boolean): void {
    this.status.hasNewGrade = hasNew;
    if (hasNew) {
      this.status.lastGradeUpdate = new Date().toISOString();
    }
    this.saveStatus();
  }

  setNewCredit(hasNew: boolean): void {
    this.status.hasNewCredit = hasNew;
    if (hasNew) {
      this.status.lastCreditUpdate = new Date().toISOString();
    }
    this.saveStatus();
  }

  setNewExam(hasNew: boolean): void {
    this.status.hasNewExam = hasNew;
    if (hasNew) {
      this.status.lastExamUpdate = new Date().toISOString();
    }
    this.saveStatus();
  }

  clearGradeNotification(): void {
    this.status.hasNewGrade = false;
    this.saveStatus();
  }

  clearCreditNotification(): void {
    this.status.hasNewCredit = false;
    this.saveStatus();
  }

  clearExamNotification(): void {
    this.status.hasNewExam = false;
    this.saveStatus();
  }

  getStatus(): NotificationStatus {
    return {
      hasNewGrade: this.status.hasNewGrade,
      hasNewCredit: this.status.hasNewCredit,
      hasNewExam: this.status.hasNewExam,
      lastGradeUpdate: this.status.lastGradeUpdate,
      lastCreditUpdate: this.status.lastCreditUpdate,
      lastExamUpdate: this.status.lastExamUpdate
    };
  }

  hasAnyNotification(): boolean {
    return this.status.hasNewGrade || this.status.hasNewCredit || this.status.hasNewExam;
  }
}

export { NotificationManager, NotificationStatus };
