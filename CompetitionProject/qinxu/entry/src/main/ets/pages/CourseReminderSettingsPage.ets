/**
 * 课程提醒设置页面模块
 * 
 * 提供课程提醒功能配置，包括：
 * - 提醒开关
 * - 提醒时间设置
 * - 课前提醒
 * - 明日课程提醒
 * 
 * @module CourseReminderSettingsPage
 */

import router from '@ohos.router';
import { promptAction } from '@kit.ArkUI';
import { CourseReminderService, CourseReminderSettings, UpcomingReminder, CourseReminderInfo } from '../services/CourseReminderService';
import { LocalDataManager, CourseData } from '../utils/LocalDataManager';
import { DataStore } from '../utils/DataStore';
import { SettingsData } from '../model/SettingsData';

/**
 * 课程提醒设置页面组件
 * 
 * 配置课程提醒的各项参数
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/CourseReminderSettingsPage' });
 * ```
 */
@Entry
@Component
struct CourseReminderSettingsPage {
  /** 提醒是否启用 */
  @State reminderEnabled: boolean = false;
  /** 提前提醒分钟数 */
  @State advanceMinutes: number = 15;
  /** 是否开启课前提醒 */
  @State remindBeforeClass: boolean = true;
  /** 是否开启明日课程提醒 */
  @State remindTomorrowCourse: boolean = true;
  /** 晚间提醒时间 */
  @State nightReminderTime: string = '21:00';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 是否有通知权限 */
  @State hasNotificationPermission: boolean = false;
  /** 活跃提醒数量 */
  @State activeRemindersCount: number = 0;
  /** 最大提醒限制 */
  @State maxRemindersLimit: number = 30;
  /** 显示提前时间选择器 */
  @State showAdvancePicker: boolean = false;
  /** 显示时间选择器 */
  @State showTimePicker: boolean = false;
  /** 临时提前分钟数 */
  @State tempAdvanceMinutes: number = 15;
  /** 临时小时 */
  @State tempHour: number = 21;
  /** 临时分钟 */
  @State tempMinute: number = 0;
  /** 提醒摘要统计 */
  @State remindersSummary: { total: number; beforeClass: number; morning: number; tomorrow: number; remaining: number } = { total: 0, beforeClass: 0, morning: 0, tomorrow: 0, remaining: 30 };
  /** 即将到来的提醒列表 */
  @State upcomingReminders: UpcomingReminder[] = [];
  /** 显示即将提醒列表 */
  @State showUpcomingReminders: boolean = false;

  /** 课程提醒服务实例 */
  private reminderService: CourseReminderService = CourseReminderService.getInstance();

  async aboutToAppear() {
    await this.loadSettings();
    await this.checkPermission();
    this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
    this.maxRemindersLimit = this.reminderService.getMaxRemindersLimit();
    this.remindersSummary = this.reminderService.getRemindersSummary();
    this.upcomingReminders = this.reminderService.getUpcomingReminders();
    this.isLoading = false;
  }

  private async loadSettings(): Promise<void> {
    try {
      const settings = this.reminderService.getSettings();
      this.reminderEnabled = settings.enabled;
      this.advanceMinutes = settings.advanceMinutes;
      this.remindBeforeClass = settings.remindBeforeClass;
      this.remindTomorrowCourse = settings.remindTomorrowCourse;
      this.nightReminderTime = settings.nightReminderTime;
    } catch (error) {
      console.error('Failed to load reminder settings:', error);
    }
  }

  private async checkPermission(): Promise<void> {
    this.hasNotificationPermission = await this.reminderService.hasNotificationPermission();
  }

  private async saveSettings(): Promise<void> {
    try {
      const previousSettings = this.reminderService.getSettings();
      const wasEnabled = previousSettings.enabled;
      const wasRemindBeforeClass = previousSettings.remindBeforeClass;
      const wasRemindTomorrowCourse = previousSettings.remindTomorrowCourse;
      
      await this.reminderService.updateSettings({
        enabled: this.reminderEnabled,
        advanceMinutes: this.advanceMinutes,
        remindBeforeClass: this.remindBeforeClass,
        remindTomorrowCourse: this.remindTomorrowCourse,
        nightReminderTime: this.nightReminderTime
      });
      
      if (!this.reminderEnabled) {
        await this.reminderService.cancelAllReminders();
        this.activeRemindersCount = 0;
        this.remindersSummary = this.reminderService.getRemindersSummary();
        this.upcomingReminders = this.reminderService.getUpcomingReminders();
        promptAction.showToast({ message: '已关闭课程提醒' });
        return;
      }
      
      if (!this.remindBeforeClass && wasRemindBeforeClass) {
        const cancelled = await this.reminderService.cancelBeforeClassReminders();
        if (cancelled > 0) {
          promptAction.showToast({ message: `已取消 ${cancelled} 个课前提醒` });
        }
        this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
        this.remindersSummary = this.reminderService.getRemindersSummary();
        this.upcomingReminders = this.reminderService.getUpcomingReminders();
      }
      
      if (!this.remindTomorrowCourse && wasRemindTomorrowCourse) {
        const cancelled = await this.reminderService.cancelTomorrowReminders();
        if (cancelled > 0) {
          promptAction.showToast({ message: `已取消 ${cancelled} 个明日课程提醒` });
        }
        this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
        this.remindersSummary = this.reminderService.getRemindersSummary();
        this.upcomingReminders = this.reminderService.getUpcomingReminders();
      }
      
      if (this.remindBeforeClass || this.remindTomorrowCourse) {
        await this.setupRemindersFromLocalData();
      }
      
      this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
      this.remindersSummary = this.reminderService.getRemindersSummary();
      this.upcomingReminders = this.reminderService.getUpcomingReminders();
      
      AppStorage.SetOrCreate('reminderSettingsChanged', true);
      console.log('CourseReminderSettingsPage: Set reminderSettingsChanged flag to true');
      
      if (this.reminderEnabled && this.activeRemindersCount > 0) {
        promptAction.showToast({ 
          message: `已设置课程提醒 ${this.activeRemindersCount} 个`,
          duration: 2000
        });
      } else if (this.reminderEnabled && this.activeRemindersCount === 0) {
        promptAction.showToast({ message: '提醒设置失败，请检查课表数据' });
      }
    } catch (error) {
      console.error('Failed to save reminder settings:', error);
    }
  }

  private async setupRemindersFromLocalData(): Promise<void> {
    try {
      console.info('CourseReminderSettingsPage: setupRemindersFromLocalData called');
      
      await this.reminderService.cleanupAllExpiredReminders();
      console.info('CourseReminderSettingsPage: Cleaned up expired reminders');
      
      const localDataManager = LocalDataManager.getInstance();
      console.info('CourseReminderSettingsPage: LocalDataManager instance obtained');
      
      const scheduleData = localDataManager.getLatestScheduleData();
      
      console.info('CourseReminderSettingsPage: scheduleData:', scheduleData ? 'exists' : 'null');
      if (scheduleData) {
        console.info('CourseReminderSettingsPage: semester:', scheduleData.semester);
        console.info('CourseReminderSettingsPage: courses count:', scheduleData.courses?.length || 0);
        if (scheduleData.courses && scheduleData.courses.length > 0) {
          console.info('CourseReminderSettingsPage: first course:', JSON.stringify(scheduleData.courses[0]));
        }
      }
      
      if (!scheduleData || !scheduleData.courses || scheduleData.courses.length === 0) {
        console.warn('CourseReminderSettingsPage: No schedule data found');
        promptAction.showToast({ message: '请先加载课表数据' });
        return;
      }

      const settings = this.reminderService.getSettings();
      const currentWeek = await this.getCurrentWeekFromSettings();
      
      console.info('CourseReminderSettingsPage: currentWeek:', currentWeek);
      
      const weekCourses = scheduleData.courses.filter((course: CourseData) => course.week === currentWeek);
      
      console.info('CourseReminderSettingsPage: weekCourses count:', weekCourses.length);
      if (weekCourses.length > 0) {
        console.info('CourseReminderSettingsPage: first weekCourse:', JSON.stringify(weekCourses[0]));
      }
      
      const reminderInfos: CourseReminderInfo[] = [];
      for (const course of weekCourses) {
        const timeParts = course.period.split('-');
        let startTime = '';
        let endTime = '';
        
        if (timeParts.length === 2) {
          const startSlot = parseInt(timeParts[0]);
          const endSlot = parseInt(timeParts[1]);
          startTime = this.getTimeBySlot(startSlot);
          endTime = this.getTimeBySlotEnd(endSlot);
        }

        if (!startTime || !endTime) {
          console.warn('CourseReminderSettingsPage: Skipping course with invalid time:', course.name, 'period:', course.period);
          continue;
        }

        console.info('CourseReminderSettingsPage: course:', course.name, 'period:', course.period, 'startTime:', startTime, 'endTime:', endTime);

        reminderInfos.push({
          courseName: course.name,
          location: course.location,
          teacher: course.teacher,
          dayOfWeek: course.dayOfWeek,
          startTime: startTime,
          endTime: endTime,
          week: course.week,
          courseType: course.courseType
        });
      }

      // 获取下一周课程数据（用于跨周明日提醒）
      const nextWeek = currentWeek + 1;
      const nextWeekCourses = scheduleData.courses.filter((course: CourseData) => course.week === nextWeek);
      const nextWeekReminderInfos: CourseReminderInfo[] = [];
      for (const course of nextWeekCourses) {
        const timeParts = course.period.split('-');
        let startTime = '';
        let endTime = '';
        
        if (timeParts.length === 2) {
          const startSlot = parseInt(timeParts[0]);
          const endSlot = parseInt(timeParts[1]);
          startTime = this.getTimeBySlot(startSlot);
          endTime = this.getTimeBySlotEnd(endSlot);
        }

        if (!startTime || !endTime) {
          continue;
        }

        nextWeekReminderInfos.push({
          courseName: course.name,
          location: course.location,
          teacher: course.teacher,
          dayOfWeek: course.dayOfWeek,
          startTime: startTime,
          endTime: endTime,
          week: course.week,
          courseType: course.courseType
        });
      }

      console.info('CourseReminderSettingsPage: reminderInfos count:', reminderInfos.length,
        'nextWeekReminderInfos count:', nextWeekReminderInfos.length);
      
      if (reminderInfos.length > 0) {
        await this.reminderService.publishWeeklyReminders(reminderInfos, nextWeekReminderInfos);
        console.info(`CourseReminderSettingsPage: Set up ${reminderInfos.length} reminders`);
        
        this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
        this.remindersSummary = this.reminderService.getRemindersSummary();
        this.upcomingReminders = this.reminderService.getUpcomingReminders();
        
        console.info('CourseReminderSettingsPage: activeRemindersCount after publish:', this.activeRemindersCount);
      }
    } catch (error) {
      console.error('CourseReminderSettingsPage: Failed to setup reminders:', error);
    }
  }

  private async getCurrentWeekFromSettings(): Promise<number> {
    try {
      const dataStore = DataStore.getInstance();
      const settings = await dataStore.loadSettings();
      console.info('CourseReminderSettingsPage: loadSettings result:', settings ? 'exists' : 'null');
      if (settings) {
        const week = settings.getCurrentWeek();
        console.info('CourseReminderSettingsPage: getCurrentWeek:', week);
        return week;
      }
      
      const defaultSettings = new SettingsData();
      const defaultWeek = defaultSettings.getCurrentWeek();
      console.info('CourseReminderSettingsPage: using default week:', defaultWeek);
      return defaultWeek;
    } catch (error) {
      console.error('CourseReminderSettingsPage: Failed to get current week:', error);
      return 1;
    }
  }

  private getTimeBySlot(slot: number): string {
    const timeSlots: Record<number, string> = {
      1: '08:30',
      2: '09:20',
      3: '10:25',
      4: '11:15',
      5: '14:30',
      6: '15:20',
      7: '16:15',
      8: '17:05',
      9: '18:20',
      10: '19:10',
      11: '20:05',
      12: '20:55'
    };
    return timeSlots[slot] || '08:30';
  }

  private getTimeBySlotEnd(slot: number): string {
    const timeSlots: Record<number, string> = {
      1: '09:15',
      2: '10:05',
      3: '11:05',
      4: '12:00',
      5: '15:15',
      6: '16:05',
      7: '17:00',
      8: '17:50',
      9: '19:05',
      10: '19:55',
      11: '20:50',
      12: '21:40'
    };
    return timeSlots[slot] || '09:15';
  }

  private async requestPermission(): Promise<void> {
    const granted = await this.reminderService.requestNotificationPermission();
    this.hasNotificationPermission = granted;
    if (granted) {
      promptAction.showToast({ message: '通知权限已开启' });
    } else {
      promptAction.showToast({ message: '请前往系统设置开启通知权限' });
    }
  }

  build() {
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
          Text('加载中...')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 12 })
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else {
        Scroll() {
          Column({ space: 12 }) {
            if (!this.hasNotificationPermission) {
              this.buildPermissionCard()
            }

            this.buildSectionHeader('提醒设置')
            this.buildToggleItem('开启课程提醒', this.reminderEnabled, async (value: boolean) => {
              if (value && !this.hasNotificationPermission) {
                await this.requestPermission();
                if (!this.hasNotificationPermission) {
                  promptAction.showToast({ message: '请先开启通知权限' });
                  return;
                }
              }
              this.reminderEnabled = value;
              await this.saveSettings();
            })

            if (this.reminderEnabled) {
              this.buildRemindersSummaryCard()

              this.buildSectionHeader('提醒时间')
              this.buildAdvanceTimeItem()

              this.buildSectionHeader('提醒方式')
              this.buildToggleItem('课前提醒', this.remindBeforeClass, async (value: boolean) => {
                this.remindBeforeClass = value;
                await this.saveSettings();
              })

              this.buildToggleItem('明日课程提醒', this.remindTomorrowCourse, async (value: boolean) => {
                this.remindTomorrowCourse = value;
                await this.saveSettings();
              })

              if (this.remindTomorrowCourse) {
                this.buildNightTimeItem()
              }

              if (this.upcomingReminders.length > 0) {
                this.buildSectionHeader('即将触发的提醒')
                this.buildUpcomingRemindersCard()
              }
              
              this.buildActionItem('清除所有提醒', () => {
                this.clearAllReminders();
              })
            }

            this.buildSectionHeader('说明')
            this.buildDescriptionCard()
          }
          .width('100%')
          .padding(16)
        }
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)
        .layoutWeight(1)
      }

      if (this.showAdvancePicker) {
        this.buildAdvancePickerDialog()
      }

      if (this.showTimePicker) {
        this.buildTimePickerDialog()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildHeader() {
    Row() {
      Button({ type: ButtonType.Circle }) {
        Image($r('app.media.chevron_left'))
          .width(24)
          .height(24)
          .fillColor($r('app.color.text_primary'))
      }
      .width(36)
      .height(36)
      .backgroundColor('transparent')
      .onClick(() => router.back())

      Text('课程提醒设置')
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
        .textAlign(TextAlign.Center)

      Blank()
        .width(36)
    }
    .width('100%')
    .padding({ left: 8, right: 8, top: 8, bottom: 8 })
    .backgroundColor($r('app.color.background_primary'))
  }

  @Builder
  buildSectionHeader(title: string) {
    Text(title)
      .fontSize(14)
      .fontColor($r('app.color.text_secondary'))
      .width('100%')
      .padding({ left: 4, bottom: 8, top: 8 })
  }

  @Builder
  buildToggleItem(title: string, value: boolean, onChange: (value: boolean) => void) {
    Row() {
      Text(title)
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Toggle({ type: ToggleType.Switch, isOn: value })
        .selectedColor($r('app.color.primary'))
        .onChange((isOn: boolean) => {
          onChange(isOn);
        })
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildActionItem(title: string, action: () => void) {
    Row() {
      Text(title)
        .fontSize(16)
        .fontColor('#f5222d')
        .layoutWeight(1)

      Image($r('app.media.chevron_right'))
        .width(16)
        .height(16)
        .fillColor($r('app.color.text_tertiary'))
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
    .onClick(action)
  }

  @Builder
  buildAdvanceTimeItem() {
    Row() {
      Text('提前提醒时间')
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Row({ space: 4 }) {
        Text(`${this.advanceMinutes} 分钟`)
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
        Image($r('app.media.chevron_right'))
          .width(16)
          .height(16)
          .fillColor($r('app.color.text_tertiary'))
      }
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
    .onClick(() => {
      this.tempAdvanceMinutes = this.advanceMinutes;
      this.showAdvancePicker = true;
    })
  }

  @Builder
  buildNightTimeItem() {
    Row() {
      Text('夜间提醒时间')
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Row({ space: 4 }) {
        Text(this.nightReminderTime)
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
        Image($r('app.media.chevron_right'))
          .width(16)
          .height(16)
          .fillColor($r('app.color.text_tertiary'))
      }
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
    .onClick(() => {
      const parts = this.nightReminderTime.split(':');
      this.tempHour = parseInt(parts[0]);
      this.tempMinute = parseInt(parts[1]);
      this.showTimePicker = true;
    })
  }

  @Builder
  buildRemindersSummaryCard() {
    Column({ space: 12 }) {
      Row() {
        Text('提醒统计')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
        
        Blank()
        
        Text(`${this.remindersSummary.total}/${this.maxRemindersLimit}`)
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.remindersSummary.remaining > 5 ? $r('app.color.primary') : '#ff9500')
      }
      .width('100%')

      Row({ space: 16 }) {
        Column() {
          Text(`${this.remindersSummary.beforeClass}`)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.primary'))
          Text('课前提醒')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 4 })
        }
        .layoutWeight(1)
        .padding(12)
        .backgroundColor($r('app.color.background_tertiary'))
        .borderRadius(12)

        Column() {
          Text(`${this.remindersSummary.tomorrow}`)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.primary'))
          Text('明日提醒')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 4 })
        }
        .layoutWeight(1)
        .padding(12)
        .backgroundColor($r('app.color.background_tertiary'))
        .borderRadius(12)
      }
      .width('100%')

      if (this.remindersSummary.remaining <= 5 && this.remindersSummary.remaining > 0) {
        Row({ space: 8 }) {
          Image($r('app.media.exclamationmark_triangle_fill'))
            .width(16)
            .height(16)
            .fillColor('#ff9500')
          Text(`剩余容量：${this.remindersSummary.remaining} 个，建议清理部分提醒`)
            .fontSize(13)
            .fontColor('#ff9500')
        }
        .width('100%')
        .padding(12)
        .backgroundColor('#fff3e0')
        .borderRadius(8)
      }
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildUpcomingRemindersCard() {
    Column({ space: 8 }) {
      ForEach(this.upcomingReminders.slice(0, 5), (reminder: UpcomingReminder, index: number) => {
        Row() {
          Column({ space: 2 }) {
            Text(reminder.courseName)
              .fontSize(14)
              .fontColor($r('app.color.text_primary'))
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
            
            Text(this.formatTriggerTime(reminder.triggerTime))
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)

          Text(this.getReminderTypeLabel(reminder.type))
            .fontSize(11)
            .fontColor($r('app.color.background_primary'))
            .backgroundColor(this.getReminderTypeColor(reminder.type))
            .borderRadius(4)
            .padding({ left: 8, right: 8, top: 4, bottom: 4 })
        }
        .width('100%')
        .padding({ top: 8, bottom: 8 })
        
        if (index < Math.min(this.upcomingReminders.length, 5) - 1) {
          Divider()
            .strokeWidth(0.5)
            .color($r('app.color.divider'))
        }
      })
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  private formatTriggerTime(date: Date): string {
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays}天后 ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    } else if (diffHours > 0) {
      return `${diffHours}小时后`;
    } else if (diffMins > 0) {
      return `${diffMins}分钟后`;
    }
    return '即将触发';
  }

  private getReminderTypeLabel(type: string): string {
    switch (type) {
      case 'before_class':
        return '课前';
      case 'morning':
        return '早晨';
      case 'tomorrow':
        return '明日';
      default:
        return '提醒';
    }
  }

  private getReminderTypeColor(type: string): ResourceStr {
    switch (type) {
      case 'before_class':
        return $r('app.color.primary');
      case 'morning':
        return '#48bb78';
      case 'tomorrow':
        return '#9f7aea';
      default:
        return $r('app.color.text_secondary');
    }
  }

  @Builder
  buildPermissionCard() {
    Column({ space: 12 }) {
      Row({ space: 12 }) {
        Image($r('app.media.bell_fill'))
          .width(24)
          .height(24)
          .fillColor('#ff9500')

        Column({ space: 4 }) {
          Text('需要通知权限')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.text_primary'))
          Text('请开启通知权限以接收课程提醒')
            .fontSize(13)
            .fontColor($r('app.color.text_secondary'))
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')

      Button('开启通知权限')
        .width('100%')
        .height(44)
        .fontSize(15)
        .fontColor($r('app.color.background_primary'))
        .backgroundColor($r('app.color.primary'))
        .borderRadius(12)
        .onClick(() => this.requestPermission())
    }
    .width('100%')
    .padding(16)
    .backgroundColor('#fff3e0')
    .borderRadius(16)
  }

  @Builder
  buildDescriptionCard() {
    Column({ space: 8 }) {
      Text('功能说明')
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      Text('• 课前提醒：在课程开始前指定时间发送提醒通知')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')

      Text('• 明日课程提醒：在夜间指定时间提醒明天的课程安排')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')

      Text('• 提醒会在应用后台运行或关闭后由系统代理发送')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')

      Text(`• 系统限制：每个应用最多设置 ${this.maxRemindersLimit} 个提醒`)
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')

      Text('• 通知会显示应用图标，点击可跳转到应用')
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildAdvancePickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('选择提前提醒时间')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        Scroll() {
          Column({ space: 10 }) {
            ForEach([5, 10, 15, 20, 30, 45, 60], (minutes: number) => {
              Row() {
                Text(`${minutes} 分钟`)
                  .fontSize(15)
                  .fontColor(this.tempAdvanceMinutes === minutes ? $r('app.color.background_primary') : $r('app.color.text_primary'))
                  .fontWeight(this.tempAdvanceMinutes === minutes ? FontWeight.Bold : FontWeight.Medium)
                  .layoutWeight(1)
              }
              .width('100%')
              .height(50)
              .padding({ left: 16, right: 16 })
              .borderRadius(12)
              .backgroundColor(this.tempAdvanceMinutes === minutes ? $r('app.color.primary') : $r('app.color.background_tertiary'))
              .onClick(() => {
                this.tempAdvanceMinutes = minutes;
              })
            })
          }
          .width('100%')
          .padding({ top: 6, bottom: 6 })
        }
        .width('100%')
        .height(280)
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)

        Row({ space: 16 }) {
          Button('取消')
            .layoutWeight(1)
            .height(44)
            .fontSize(15)
            .fontColor($r('app.color.text_primary'))
            .backgroundColor($r('app.color.background_tertiary'))
            .borderRadius(12)
            .onClick(() => {
              this.showAdvancePicker = false;
            })

          Button('确定')
            .layoutWeight(1)
            .height(44)
            .fontSize(15)
            .fontColor($r('app.color.background_primary'))
            .backgroundColor($r('app.color.primary'))
            .borderRadius(12)
            .onClick(async () => {
              this.advanceMinutes = this.tempAdvanceMinutes;
              await this.saveSettings();
              this.showAdvancePicker = false;
              promptAction.showToast({ message: `提前提醒时间已设置为 ${this.advanceMinutes} 分钟` });
            })
        }
        .width('100%')
      }
      .width('80%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(20)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor($r('app.color.overlay'))
    .onClick(() => {
      this.showAdvancePicker = false;
    })
  }

  @Builder
  buildTimePickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('选择夜间提醒时间')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        TimePicker({
          selected: new Date(2024, 0, 1, this.tempHour, this.tempMinute)
        })
          .useMilitaryTime(true)
          .disappearTextStyle({ color: $r('app.color.text_tertiary'), font: { size: 14 } })
          .textStyle({ color: $r('app.color.text_primary'), font: { size: 16 } })
          .selectedTextStyle({ color: $r('app.color.primary'), font: { size: 18, weight: FontWeight.Bold } })
          .onChange((value: TimePickerResult) => {
            this.tempHour = value.hour as number;
            this.tempMinute = value.minute as number;
          })
          .width('100%')
          .height(180)

        Row({ space: 16 }) {
          Button('取消')
            .layoutWeight(1)
            .height(44)
            .fontSize(15)
            .fontColor($r('app.color.text_primary'))
            .backgroundColor($r('app.color.background_tertiary'))
            .borderRadius(12)
            .onClick(() => {
              this.showTimePicker = false;
            })

          Button('确定')
            .layoutWeight(1)
            .height(44)
            .fontSize(15)
            .fontColor($r('app.color.background_primary'))
            .backgroundColor($r('app.color.primary'))
            .borderRadius(12)
            .onClick(async () => {
              this.nightReminderTime = `${String(this.tempHour).padStart(2, '0')}:${String(this.tempMinute).padStart(2, '0')}`;
              await this.saveSettings();
              this.showTimePicker = false;
              promptAction.showToast({ message: `夜间提醒时间已设置为 ${this.nightReminderTime}` });
            })
        }
        .width('100%')
      }
      .width('85%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(20)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor($r('app.color.overlay'))
    .onClick(() => {
      this.showTimePicker = false;
    })
  }

  private async clearAllReminders(): Promise<void> {
    try {
      await this.reminderService.cancelAllReminders();
      this.activeRemindersCount = 0;
      this.remindersSummary = this.reminderService.getRemindersSummary();
      this.upcomingReminders = this.reminderService.getUpcomingReminders();
      promptAction.showToast({ message: '已清除所有提醒，当前有 0 个提醒' });
    } catch (error) {
      console.error('Failed to clear reminders:', error);
      promptAction.showToast({ message: '清除提醒失败' });
    }
  }
}

export { CourseReminderSettingsPage };