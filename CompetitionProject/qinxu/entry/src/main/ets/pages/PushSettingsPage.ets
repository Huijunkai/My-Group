/**
 * 推送设置页面模块
 * 
 * 提供推送通知设置功能，包括：
 * - 课程变更通知
 * - 成绩更新通知
 * - 考试安排通知
 * - 公告通知
 * 
 * @module PushSettingsPage
 */

import { PushNotificationService, PushNotificationSettings } from '../services/PushNotificationService';
import { ApiClient } from '../utils/ApiClient';
import router from '@ohos.router';

/**
 * 推送设置页面组件
 * 
 * 配置各类推送通知的开关
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/PushSettingsPage' });
 * ```
 */
@Entry
@Component
struct PushSettingsPage {
  /** 课程变更通知开关 */
  @State private courseChangeEnabled: boolean = true;
  /** 新成绩通知开关 */
  @State private newGradeEnabled: boolean = true;
  /** 新考试通知开关 */
  @State private newExamEnabled: boolean = true;
  /** 考试提醒开关 */
  @State private examReminderEnabled: boolean = true;
  /** 公告通知开关 */
  @State private announcementEnabled: boolean = true;
  /** 推送Token */
  @State private pushToken: string = '';
  /** 是否已注册 */
  @State private isRegistered: boolean = false;
  /** 加载状态 */
  @State private isLoading: boolean = false;

  /** 推送服务实例 */
  private pushService: PushNotificationService = PushNotificationService.getInstance();

  aboutToAppear(): void {
    this.loadSettings();
    this.checkRegistrationStatus();
  }

  private loadSettings(): void {
    const settings: PushNotificationSettings = this.pushService.getSettings();
    this.courseChangeEnabled = settings.courseChangeEnabled;
    this.newGradeEnabled = settings.newGradeEnabled;
    this.newExamEnabled = settings.newExamEnabled;
    this.examReminderEnabled = settings.examReminderEnabled;
    this.announcementEnabled = settings.announcementEnabled;
    this.isRegistered = settings.isPushEnabled;
    this.pushToken = this.pushService.getPushTokenForServer();
  }

  private async checkRegistrationStatus(): Promise<void> {
    const studentId = AppStorage.Get('studentId') as string;
    if (studentId && this.pushToken) {
      this.isRegistered = true;
    }
  }

  private async saveSettings(): Promise<void> {
    const settings: PushNotificationSettings = {
      courseChangeEnabled: this.courseChangeEnabled,
      newGradeEnabled: this.newGradeEnabled,
      newExamEnabled: this.newExamEnabled,
      examReminderEnabled: this.examReminderEnabled,
      announcementEnabled: this.announcementEnabled,
      isPushEnabled: this.isRegistered
    };
    await this.pushService.updateSettings(settings);
  }

  private async registerPushToken(): Promise<void> {
    const studentId = AppStorage.Get('studentId') as string;
    if (!studentId) {
      console.error('PushSettingsPage: No student ID found');
      return;
    }

    this.isLoading = true;
    try {
      const token = await this.pushService.getPushToken();
      if (!token) {
        console.error('PushSettingsPage: Failed to get push token');
        return;
      }

      const response = await ApiClient.registerPushToken(studentId, token);
      if (response.success) {
        this.isRegistered = true;
        this.pushToken = token;
        await this.pushService.bindUserAccount(studentId);
        await this.pushService.setPushEnabled(true);
        console.info('PushSettingsPage: Push token registered successfully');
      }
    } catch (error) {
      console.error('PushSettingsPage: Failed to register push token:', error);
    } finally {
      this.isLoading = false;
    }
  }

  private async unregisterPushToken(): Promise<void> {
    const studentId = AppStorage.Get('studentId') as string;
    if (!studentId) return;

    this.isLoading = true;
    try {
      await ApiClient.unregisterPushToken(studentId);
      this.isRegistered = false;
      await this.pushService.setPushEnabled(false);
      console.info('PushSettingsPage: Push token unregistered');
    } catch (error) {
      console.error('PushSettingsPage: Failed to unregister push token:', error);
    } finally {
      this.isLoading = false;
    }
  }

  build() {
    Column() {
      Column()
        .width('100%')
        .height(44)

      Row() {
        Image($r('app.media.chevron_left'))
          .width(24)
          .height(24)
          .fillColor($r('app.color.text_primary'))
          .onClick(() => router.back())

        Text('推送通知设置')
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Blank().width(24)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      Scroll() {
        Column({ space: 16 }) {
          Column() {
            Row() {
              Text('推送服务状态')
                .fontSize(16)
                .fontWeight(FontWeight.Medium)

              Blank()

              if (this.isRegistered) {
                Text('已启用')
                  .fontSize(14)
                  .fontColor('#4CAF50')
              } else {
                Text('未启用')
                  .fontSize(14)
                  .fontColor('#FF9800')
              }
            }
            .width('100%')
            .padding(16)

            if (this.pushToken) {
              Text('Token: ' + this.pushToken.substring(0, 20) + '...')
                .fontSize(12)
                .fontColor($r('app.color.text_tertiary'))
                .width('100%')
                .padding({ left: 16, right: 16, bottom: 16 })
            }

            Button(this.isRegistered ? '关闭推送服务' : '开启推送服务')
              .width('90%')
              .height(44)
              .backgroundColor(this.isRegistered ? '#F44336' : '#2196F3')
              .fontColor(Color.White)
              .margin({ bottom: 16 })
              .enabled(!this.isLoading)
              .onClick(() => {
                if (this.isRegistered) {
                  this.unregisterPushToken();
                } else {
                  this.registerPushToken();
                }
              })
          }
          .width('100%')
          .backgroundColor($r('app.color.background_primary'))
          .borderRadius(16)

          Column() {
            Text('通知类型')
              .fontSize(16)
              .fontWeight(FontWeight.Medium)
              .width('100%')
              .padding(16)

            Row() {
              Column() {
                Text('课程变动通知')
                  .fontSize(15)
                  .fontColor($r('app.color.text_primary'))

                Text('当课表发生变化时通知您')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .margin({ top: 4 })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)

              Toggle({ type: ToggleType.Switch, isOn: this.courseChangeEnabled })
                .selectedColor('#2196F3')
                .onChange((value: boolean) => {
                  this.courseChangeEnabled = value;
                  this.saveSettings();
                })
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })

            Row() {
              Column() {
                Text('新成绩通知')
                  .fontSize(15)
                  .fontColor($r('app.color.text_primary'))

                Text('当有新成绩发布时通知您')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .margin({ top: 4 })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)

              Toggle({ type: ToggleType.Switch, isOn: this.newGradeEnabled })
                .selectedColor('#2196F3')
                .onChange((value: boolean) => {
                  this.newGradeEnabled = value;
                  this.saveSettings();
                })
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })

            Row() {
              Column() {
                Text('新考试通知')
                  .fontSize(15)
                  .fontColor($r('app.color.text_primary'))

                Text('当有新考试安排时通知您')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .margin({ top: 4 })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)

              Toggle({ type: ToggleType.Switch, isOn: this.newExamEnabled })
                .selectedColor('#2196F3')
                .onChange((value: boolean) => {
                  this.newExamEnabled = value;
                  this.saveSettings();
                })
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })

            Row() {
              Column() {
                Text('考试提醒')
                  .fontSize(15)
                  .fontColor($r('app.color.text_primary'))

                Text('考试前24小时提醒您')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .margin({ top: 4 })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)

              Toggle({ type: ToggleType.Switch, isOn: this.examReminderEnabled })
                .selectedColor('#2196F3')
                .onChange((value: boolean) => {
                  this.examReminderEnabled = value;
                  this.saveSettings();
                })
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })

            Row() {
              Column() {
                Text('教务公告通知')
                  .fontSize(15)
                  .fontColor($r('app.color.text_primary'))

                Text('重修、补考、体质健康测试、选课等通知')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .margin({ top: 4 })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)

              Toggle({ type: ToggleType.Switch, isOn: this.announcementEnabled })
                .selectedColor('#2196F3')
                .onChange((value: boolean) => {
                  this.announcementEnabled = value;
                  this.saveSettings();
                })
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })
          }
          .width('100%')
          .backgroundColor($r('app.color.background_primary'))
          .borderRadius(16)

          Column() {
            Text('说明')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
              .width('100%')
              .padding(16)

            Text('• 推送服务需要网络连接')
              .fontSize(12)
              .fontColor($r('app.color.text_tertiary'))
              .width('100%')
              .padding({ left: 16, right: 16 })

            Text('• 成绩和考试通知由服务器定时检测')
              .fontSize(12)
              .fontColor($r('app.color.text_tertiary'))
              .width('100%')
              .padding({ left: 16, right: 16 })

            Text('• 课程变动通知在同步数据时检测')
              .fontSize(12)
              .fontColor($r('app.color.text_tertiary'))
              .width('100%')
              .padding({ left: 16, right: 16 })

            Text('• 公告通知包含重修、补考、体质健康测试、选课、补修、免修等关键词')
              .fontSize(12)
              .fontColor($r('app.color.text_tertiary'))
              .width('100%')
              .padding({ left: 16, right: 16, bottom: 16 })
          }
          .width('100%')
          .backgroundColor($r('app.color.background_primary'))
          .borderRadius(16)
        }
        .padding(16)
      }
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
  }
}
