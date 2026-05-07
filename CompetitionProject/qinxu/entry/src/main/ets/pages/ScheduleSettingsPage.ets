/**
 * 课表设置页面模块
 * 
 * 提供课表相关设置功能，包括：
 * - 学期信息设置
 * - 开学日期设置
 * - 总周数设置
 * - 课程提醒设置
 * 
 * @module ScheduleSettingsPage
 */

import router from '@ohos.router';
import { promptAction } from '@kit.ArkUI';
import { DataStore } from '../utils/DataStore';
import { SettingsData } from '../model/SettingsData';
import { CourseReminderService, CourseReminderSettings } from '../services/CourseReminderService';

/**
 * 课表设置页面组件
 * 
 * 配置学期信息和课程提醒参数
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/ScheduleSettingsPage' });
 * ```
 */
@Entry
@Component
struct ScheduleSettingsPage {
  /** 学期名称 */
  @State semesterName: string = '';
  /** 开学日期 */
  @State semesterStartDate: string = '';
  /** 总周数 */
  @State totalWeeks: number = 20;
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 显示日期选择器 */
  @State showDatePicker: boolean = false;
  /** 显示周数选择器 */
  @State showWeekPicker: boolean = false;
  /** 临时开始日期 */
  @State tempStartDate: string = '';
  /** 临时周数 */
  @State tempWeeks: number = 20;
  /** 临时年份 */
  @State tempYear: number = new Date().getFullYear();
  /** 临时月份 */
  @State tempMonth: number = new Date().getMonth() + 1;
  /** 临时日期 */
  @State tempDay: number = 1;

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
  @State tempHour: number = 7;
  /** 临时分钟 */
  @State tempMinute: number = 30;

  /** 数据存储实例 */
  private dataStore: DataStore = DataStore.getInstance();
  /** 课程提醒服务实例 */
  private reminderService: CourseReminderService = CourseReminderService.getInstance();

  async aboutToAppear() {
    await this.loadSettings();
    await this.loadReminderSettings();
    await this.checkPermission();
    this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
    this.maxRemindersLimit = this.reminderService.getMaxRemindersLimit();
    this.isLoading = false;
  }

  private async loadSettings(): Promise<void> {
    try {
      const loadedSettings: SettingsData | null = await this.dataStore.loadSettings();
      if (loadedSettings) {
        this.semesterName = loadedSettings.semesterName;
        this.semesterStartDate = loadedSettings.semesterStartDate;
        this.totalWeeks = loadedSettings.totalWeeks;
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }

  private async loadReminderSettings(): Promise<void> {
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

  private async saveSettings(): Promise<boolean> {
    try {
      const settings = new SettingsData();
      settings.semesterName = this.semesterName;
      settings.semesterStartDate = this.semesterStartDate;
      settings.totalWeeks = this.totalWeeks;
      
      await this.dataStore.saveSettings(settings);
      AppStorage.SetOrCreate('scheduleSettingsChanged', true);
      return true;
    } catch (error) {
      console.error('Failed to save settings:', error);
      return false;
    }
  }

  private async saveReminderSettings(): Promise<void> {
    try {
      await this.reminderService.updateSettings({
        enabled: this.reminderEnabled,
        advanceMinutes: this.advanceMinutes,
        remindBeforeClass: this.remindBeforeClass,
        remindTomorrowCourse: this.remindTomorrowCourse,
        nightReminderTime: this.nightReminderTime
      });
      
      this.activeRemindersCount = this.reminderService.getActiveRemindersCount();
      
      if (!this.remindTomorrowCourse) {
        await this.reminderService.cancelTomorrowReminders();
      }
      
      AppStorage.SetOrCreate('reminderSettingsChanged', true);
    } catch (error) {
      console.error('Failed to save reminder settings:', error);
    }
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
            this.buildSectionHeader('学期设置')
            this.buildSettingItem('学期名称', this.formatSemesterDisplay(this.semesterName), () => {
              promptAction.showToast({ message: '学期名称由系统自动获取' });
            })
            this.buildDateItem()
            this.buildWeeksItem()

            this.buildSectionHeader('课程提醒')
            if (!this.hasNotificationPermission) {
              this.buildPermissionCard()
            }
            this.buildToggleItem('开启课程提醒', this.reminderEnabled, async (value: boolean) => {
              if (value && !this.hasNotificationPermission) {
                await this.requestPermission();
                if (!this.hasNotificationPermission) {
                  promptAction.showToast({ message: '请先开启通知权限' });
                  return;
                }
              }
              this.reminderEnabled = value;
              await this.saveReminderSettings();
              if (value) {
                promptAction.showToast({ message: '课程提醒已开启' });
              } else {
                promptAction.showToast({ message: '课程提醒已关闭' });
              }
            })

            if (this.reminderEnabled) {
              this.buildRemindersCountCard()
              this.buildAdvanceTimeItem()
              this.buildToggleItem('课前提醒', this.remindBeforeClass, async (value: boolean) => {
                this.remindBeforeClass = value;
                await this.saveReminderSettings();
              })

              this.buildToggleItem('明日课程提醒', this.remindTomorrowCourse, async (value: boolean) => {
                this.remindTomorrowCourse = value;
                await this.saveReminderSettings();
              })

              if (this.remindTomorrowCourse) {
                this.buildNightTimeItem()
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

      if (this.showDatePicker) {
        this.buildDatePickerDialog()
      }

      if (this.showWeekPicker) {
        this.buildWeekPickerDialog()
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

      Text('课表设置')
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
  buildSettingItem(title: string, value: string, action: () => void) {
    Row() {
      Text(title)
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Row({ space: 4 }) {
        Text(value)
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
    .onClick(action)
  }

  @Builder
  buildDateItem() {
    Row() {
      Text('开学日期')
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Row({ space: 4 }) {
        Text(this.formatDate(this.semesterStartDate))
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
      if (this.semesterStartDate) {
        const date = new Date(this.semesterStartDate);
        this.tempYear = date.getFullYear();
        this.tempMonth = date.getMonth() + 1;
        this.tempDay = date.getDate();
        this.tempStartDate = this.semesterStartDate;
      } else {
        const today = new Date();
        this.tempYear = today.getFullYear();
        this.tempMonth = today.getMonth() + 1;
        this.tempDay = today.getDate();
        this.tempStartDate = '';
      }
      this.showDatePicker = true;
    })
  }

  @Builder
  buildWeeksItem() {
    Row() {
      Text('总周数')
        .fontSize(16)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Row({ space: 4 }) {
        Text(`${this.totalWeeks}周`)
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
      this.tempWeeks = this.totalWeeks;
      this.showWeekPicker = true;
    })
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
      const parts: string[] = this.nightReminderTime.split(':');
      this.tempHour = parseInt(parts[0]);
      this.tempMinute = parseInt(parts[1]);
      this.showTimePicker = true;
    })
  }

  @Builder
  buildRemindersCountCard() {
    Row({ space: 12 }) {
      Image($r('app.media.bell_fill'))
        .width(20)
        .height(20)
        .fillColor($r('app.color.primary'))

      Text('已设置提醒')
        .fontSize(15)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Text(`${this.activeRemindersCount}/${this.maxRemindersLimit}`)
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.background_primary'))
        .backgroundColor($r('app.color.primary'))
        .borderRadius(12)
        .padding({ left: 12, right: 12, top: 4, bottom: 4 })
    }
    .width('100%')
    .height(56)
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
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
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildDatePickerDialog() {
    Column() {
      Column() {
        Text('选择开学日期')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .width('100%')

        DatePicker({
          start: new Date('2020-01-01'),
          end: new Date('2030-12-31'),
          selected: this.tempStartDate ? new Date(this.tempStartDate) : new Date()
        })
          .lunar(false)
          .disappearTextStyle({ color: $r('app.color.text_tertiary'), font: { size: 14, weight: FontWeight.Normal } })
          .textStyle({ color: $r('app.color.text_primary'), font: { size: 16, weight: FontWeight.Medium } })
          .selectedTextStyle({ color: $r('app.color.primary'), font: { size: 18, weight: FontWeight.Bold } })
          .onDateChange((value: Date) => {
            this.tempYear = value.getFullYear();
            this.tempMonth = value.getMonth() + 1;
            this.tempDay = value.getDate();
          })
          .width('100%')
          .height(200)
          .margin({ top: 16 })

        Row({ space: 16 }) {
          Button('取消')
            .layoutWeight(1)
            .height(48)
            .fontSize(16)
            .fontColor($r('app.color.text_primary'))
            .backgroundColor($r('app.color.background_tertiary'))
            .borderRadius(24)
            .onClick(() => {
              this.showDatePicker = false;
            })

          Button('确定')
            .layoutWeight(1)
            .height(48)
            .fontSize(16)
            .fontColor($r('app.color.background_primary'))
            .backgroundColor($r('app.color.primary'))
            .borderRadius(24)
            .onClick(() => {
              this.confirmDateSelection();
            })
        }
        .width('100%')
        .margin({ top: 24 })
      }
      .width('85%')
      .padding(24)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(24)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor($r('app.color.overlay'))
    .onClick(() => {
      this.showDatePicker = false;
    })
  }

  @Builder
  buildWeekPickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('选择总周数')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        Scroll() {
          Column({ space: 10 }) {
            ForEach(this.getWeekArray(), (week: number) => {
              Row() {
                Text(`${week}周`)
                  .fontSize(15)
                  .fontColor(this.tempWeeks === week ? $r('app.color.background_primary') : $r('app.color.text_primary'))
                  .fontWeight(this.tempWeeks === week ? FontWeight.Bold : FontWeight.Medium)
                  .layoutWeight(1)
              }
              .width('100%')
              .height(50)
              .padding({ left: 16, right: 16 })
              .borderRadius(12)
              .backgroundColor(this.tempWeeks === week ? $r('app.color.primary') : $r('app.color.background_tertiary'))
              .onClick(() => {
                this.tempWeeks = week;
                this.confirmWeekSelection();
              })
            })
          }
          .width('100%')
          .padding({ top: 6, bottom: 6 })
        }
        .width('100%')
        .height(240)
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)
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
      this.showWeekPicker = false;
    })
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
              await this.saveReminderSettings();
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
              await this.saveReminderSettings();
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

  private async confirmWeekSelection(): Promise<void> {
    if (this.tempWeeks === this.totalWeeks) {
      this.showWeekPicker = false;
      return;
    }

    this.totalWeeks = this.tempWeeks;
    const saved = await this.saveSettings();
    this.showWeekPicker = false;
    if (saved) {
      promptAction.showToast({ message: `总周数已设置为${this.tempWeeks}周` });
    } else {
      promptAction.showToast({ message: '总周数设置失败' });
    }
  }

  private async confirmDateSelection(): Promise<void> {
    const year = this.tempYear;
    const month = String(this.tempMonth).padStart(2, '0');
    const day = String(this.tempDay).padStart(2, '0');
    this.semesterStartDate = `${year}-${month}-${day}`;
    this.tempStartDate = this.semesterStartDate;
    
    this.showDatePicker = false;
    const saved = await this.saveSettings();
    
    if (saved) {
      promptAction.showToast({ message: '开学日期已更新' });
    } else {
      promptAction.showToast({ message: '开学日期更新失败' });
    }
  }

  private async clearAllReminders(): Promise<void> {
    try {
      await this.reminderService.cancelAllReminders();
      this.activeRemindersCount = 0;
      promptAction.showToast({ message: '已清除所有提醒' });
    } catch (error) {
      console.error('Failed to clear reminders:', error);
      promptAction.showToast({ message: '清除提醒失败' });
    }
  }

  private formatDate(dateStr: string): string {
    if (!dateStr) return '未设置';
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return '未设置';
    return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
  }

  private formatSemesterDisplay(semester: string): string {
    if (!semester) return '未设置';
    if (semester.includes('-')) {
      const parts = semester.split('-');
      if (parts.length === 3) {
        const term = parts[2] === '1' ? '第一学期' : '第二学期';
        return `${parts[0]}-${parts[1]}学年 ${term}`;
      }
    }
    return semester;
  }

  private getWeekArray(): number[] {
    const weeks: number[] = [];
    const minWeeks = Math.max(1, this.totalWeeks - 5);
    const maxWeeks = this.totalWeeks + 5;
    for (let i = minWeeks; i <= maxWeeks; i++) {
      weeks.push(i);
    }
    return weeks;
  }

  aboutToDisappear() {
    this.saveSettings();
  }
}

export { ScheduleSettingsPage };
