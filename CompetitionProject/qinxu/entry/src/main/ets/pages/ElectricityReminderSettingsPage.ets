/**
 * 电费提醒设置页面模块
 * 
 * 提供电费提醒功能配置，包括：
 * - 提醒开关
 * - 余额阈值设置
 * - 宿舍信息绑定
 * 
 * @module ElectricityReminderSettingsPage
 */

import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';
import router from '@ohos.router';
import { XyyxtApiClient, XyyxtElectricityReminderSettings } from '../utils/XyyxtApiClient';

/**
 * 电费提醒设置页面组件
 * 
 * 配置电费余额提醒的各项参数
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/ElectricityReminderSettingsPage' });
 * ```
 */
@Entry
@Component
struct ElectricityReminderSettingsPage {
  /** 提醒是否启用 */
  @State isEnabled: boolean = false;
  /** 余额阈值 */
  @State threshold: string = '10';
  /** 电费账号 */
  @State electricityAccount: string = '';
  /** 电费密码 */
  @State electricityPassword: string = '';
  /** 房间名称 */
  @State roomName: string = '';
  /** 房间ID */
  @State roomId: string = '';
  /** 校区ID */
  @State campusId: string = '';
  /** 楼栋ID */
  @State buildingId: string = '';
  /** 校区名称 */
  @State campusName: string = '';
  /** 楼栋名称 */
  @State buildingName: string = '';
  /** 加载状态 */
  @State isLoading: boolean = false;
  /** 保存成功标志 */
  @State saveSuccess: boolean = false;
  /** 保存消息 */
  @State saveMessage: string = '';
  /** 是否有宿舍设置 */
  @State hasDormitorySettings: boolean = false;
  /** 学生ID（全局存储属性） */
  @StorageProp('studentId') studentId: string = '';
  /** 偏好设置存储名称 */
  private PREFERENCES_NAME = 'electricity_reminder_settings';
  /** 宿舍偏好设置存储名称 */
  private DORMITORY_PREFERENCES = 'dormitory_settings';
  /** 一信通API客户端实例 */
  private xyyxtClient: XyyxtApiClient = XyyxtApiClient.getInstance();

  aboutToAppear(): void {
    this.loadSettings();
  }

  private async loadSettings(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      
      const dormitoryPrefs = await preferences.getPreferences(context, this.DORMITORY_PREFERENCES);
      const savedUsername = await dormitoryPrefs.get('username', '') as string;
      const savedPassword = await dormitoryPrefs.get('password', '') as string;
      const savedRoomId = await dormitoryPrefs.get('roomId', '') as string;
      const savedRoomName = await dormitoryPrefs.get('roomName', '') as string;
      const savedBuildingId = await dormitoryPrefs.get('buildingId', '') as string;
      const savedBuildingName = await dormitoryPrefs.get('buildingName', '') as string;
      const savedCampusId = await dormitoryPrefs.get('campusId', '') as string;
      const savedCampusName = await dormitoryPrefs.get('campusName', '') as string;
      
      if (savedUsername && savedRoomId) {
        this.hasDormitorySettings = true;
        this.electricityAccount = savedUsername;
        this.electricityPassword = savedPassword;
        this.roomId = savedRoomId;
        this.roomName = savedRoomName;
        this.buildingId = savedBuildingId;
        this.buildingName = savedBuildingName;
        this.campusId = savedCampusId;
        this.campusName = savedCampusName;
      }
      
      const dataPreferences = await preferences.getPreferences(context, this.PREFERENCES_NAME);
      this.isEnabled = await dataPreferences.get('enabled', false) as boolean;
      const savedThreshold = await dataPreferences.get('threshold', 10);
      this.threshold = String(savedThreshold);
      
    } catch (error) {
      console.error('加载设置失败:', error);
    }
  }

  private async saveSettings(): Promise<void> {
    const thresholdNum = parseFloat(this.threshold);
    if (isNaN(thresholdNum) || thresholdNum <= 0) {
      this.saveMessage = '请输入有效的提醒金额';
      this.saveSuccess = false;
      return;
    }

    if (!this.hasDormitorySettings) {
      this.saveMessage = '请先在首页绑定宿舍信息';
      this.saveSuccess = false;
      return;
    }

    if (!this.electricityPassword) {
      this.saveMessage = '请输入校园一信通密码';
      this.saveSuccess = false;
      return;
    }

    this.isLoading = true;
    this.saveSuccess = false;
    this.saveMessage = '';

    try {
      const context = getContext(this) as common.UIAbilityContext;
      const dataPreferences = await preferences.getPreferences(context, this.PREFERENCES_NAME);
      await dataPreferences.put('enabled', this.isEnabled);
      await dataPreferences.put('threshold', thresholdNum);
      await dataPreferences.flush();

      const settings: XyyxtElectricityReminderSettings = {
        enabled: this.isEnabled,
        threshold: thresholdNum,
        electricityAccount: this.electricityAccount,
        electricityPassword: this.electricityPassword,
        roomId: this.roomId,
        campus: this.campusId,
        building: this.buildingId
      };

      const result = await this.xyyxtClient.saveElectricityReminderSettings(this.studentId, settings);
      
      if (result.success) {
        this.saveSuccess = true;
        this.saveMessage = '设置保存成功';
      } else {
        this.saveSuccess = false;
        this.saveMessage = result.message || '保存失败，请重试';
      }
    } catch (error) {
      console.error('保存设置失败:', error);
      this.saveSuccess = false;
      this.saveMessage = '保存失败: ' + (error as Error).message;
    } finally {
      this.isLoading = false;
    }
  }

  build() {
    Column() {
      Row() {
        Image($r('app.media.chevron_left'))
          .width(24)
          .height(24)
          .fillColor($r('app.color.text_primary'))
          .onClick(() => router.back())

        Text('电费提醒设置')
          .fontSize(20)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .margin({ left: 12 })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      Scroll() {
        Column({ space: 16 }) {
          if (!this.hasDormitorySettings) {
            Column({ space: 12 }) {
              Row() {
                Image($r('app.media.info_circle'))
                  .width(20)
                  .height(20)
                  .fillColor($r('app.color.warning'))
                Text('未绑定宿舍信息')
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .margin({ left: 8 })
              }
              
              Text('请先在首页"宿舍信息"卡片中登录校园一信通并选择宿舍，系统将自动获取您的电费账号和宿舍信息。')
                .fontSize(14)
                .fontColor($r('app.color.text_secondary'))
                .width('100%')
              
              Button('前往绑定')
                .width('100%')
                .height(44)
                .fontSize(15)
                .backgroundColor($r('app.color.primary'))
                .fontColor($r('app.color.button_text'))
                .borderRadius(12)
                .margin({ top: 8 })
                .onClick(() => {
                  router.back();
                })
            }
            .width('100%')
            .padding(20)
            .backgroundColor($r('app.color.warning_light'))
            .borderRadius(16)
          } else {
            Column() {
              Row() {
                Row() {
                  Image($r('app.media.bolt'))
                    .width(20)
                    .height(20)
                    .fillColor($r('app.color.primary'))
                  Text('提醒开关')
                    .fontSize(16)
                    .fontWeight(FontWeight.Medium)
                    .fontColor($r('app.color.text_primary'))
                    .margin({ left: 8 })
                }

                Toggle({ type: ToggleType.Switch, isOn: this.isEnabled })
                  .onChange((value: boolean) => {
                    this.isEnabled = value;
                  })
                  .selectedColor($r('app.color.primary'))
              }
              .width('100%')
              .justifyContent(FlexAlign.SpaceBetween)
              .padding(16)

              if (this.isEnabled) {
                Divider().color($r('app.color.divider'))

                Column({ space: 16 }) {
                  Column({ space: 8 }) {
                    Text('绑定信息')
                      .fontSize(14)
                      .fontColor($r('app.color.text_tertiary'))
                      .width('100%')
                    
                    Row() {
                      Column() {
                        Image($r('app.media.person'))
                          .width(20)
                          .height(20)
                          .fillColor($r('app.color.primary'))
                      }
                      .width(40)
                      .height(40)
                      .justifyContent(FlexAlign.Center)
                      .backgroundColor($r('app.color.info_light'))
                      .borderRadius(10)
                      
                      Column({ space: 2 }) {
                        Text('校园一信通账号')
                          .fontSize(12)
                          .fontColor($r('app.color.text_tertiary'))
                        Text(this.electricityAccount)
                          .fontSize(15)
                          .fontWeight(FontWeight.Medium)
                          .fontColor($r('app.color.text_primary'))
                      }
                      .alignItems(HorizontalAlign.Start)
                      .layoutWeight(1)
                      .margin({ left: 12 })
                    }
                    .width('100%')
                    .padding(12)
                    .backgroundColor($r('app.color.background_secondary'))
                    .borderRadius(12)
                    
                    Row() {
                      Column() {
                        Image($r('app.media.house_fill'))
                          .width(20)
                          .height(20)
                          .fillColor($r('app.color.primary'))
                      }
                      .width(40)
                      .height(40)
                      .justifyContent(FlexAlign.Center)
                      .backgroundColor($r('app.color.info_light'))
                      .borderRadius(10)
                      
                      Column({ space: 2 }) {
                        Text('宿舍房间')
                          .fontSize(12)
                          .fontColor($r('app.color.text_tertiary'))
                        Text(this.roomName || this.roomId)
                          .fontSize(15)
                          .fontWeight(FontWeight.Medium)
                          .fontColor($r('app.color.text_primary'))
                        if (this.buildingName || this.campusName) {
                          Text(`${this.campusName || ''} ${this.buildingName || ''}`)
                            .fontSize(12)
                            .fontColor($r('app.color.text_tertiary'))
                        }
                      }
                      .alignItems(HorizontalAlign.Start)
                      .layoutWeight(1)
                      .margin({ left: 12 })
                    }
                    .width('100%')
                    .padding(12)
                    .backgroundColor($r('app.color.background_secondary'))
                    .borderRadius(12)
                  }
                  .width('100%')
                  .padding({ left: 16, right: 16 })

                  Column({ space: 8 }) {
                    Text('校园一信通密码')
                      .fontSize(14)
                      .fontColor($r('app.color.text_secondary'))
                      .width('100%')
                    TextInput({ placeholder: '请输入密码用于自动查询电费', text: this.electricityPassword })
                      .type(InputType.Password)
                      .height(48)
                      .width('100%')
                      .backgroundColor($r('app.color.background_secondary'))
                      .borderRadius(10)
                      .padding({ left: 12, right: 12 })
                      .onChange((value: string) => {
                        this.electricityPassword = value;
                      })
                    Text('密码将加密存储，用于自动登录查询电费余额')
                      .fontSize(12)
                      .fontColor($r('app.color.text_tertiary'))
                  }
                  .width('100%')
                  .padding({ left: 16, right: 16 })

                  Column({ space: 8 }) {
                    Row() {
                      Text('提醒金额')
                        .fontSize(14)
                        .fontColor($r('app.color.text_secondary'))
                      Text('（元）')
                        .fontSize(14)
                        .fontColor($r('app.color.text_tertiary'))
                    }
                    TextInput({ placeholder: '请输入提醒金额', text: this.threshold })
                      .type(InputType.Number)
                      .height(48)
                      .width('100%')
                      .backgroundColor($r('app.color.background_secondary'))
                      .borderRadius(10)
                      .padding({ left: 12, right: 12 })
                      .onChange((value: string) => {
                        this.threshold = value;
                      })
                    Text('当电费余额低于此金额时，将收到提醒通知')
                      .fontSize(12)
                      .fontColor($r('app.color.text_tertiary'))
                  }
                  .width('100%')
                  .padding({ left: 16, right: 16, bottom: 16 })
                }
              }
            }
            .width('100%')
            .backgroundColor($r('app.color.card_bg'))
            .borderRadius(16)
            .margin({ top: 8 })
          }

          if (this.saveMessage) {
            Row() {
              Image(this.saveSuccess ? $r('app.media.checkmark_circle') : $r('app.media.exclamationmark_triangle_fill'))
                .width(16)
                .height(16)
                .fillColor(this.saveSuccess ? $r('app.color.success') : $r('app.color.danger'))
              Text(this.saveMessage)
                .fontSize(14)
                .fontColor(this.saveSuccess ? $r('app.color.success') : $r('app.color.danger'))
                .margin({ left: 8 })
            }
            .padding(12)
            .backgroundColor(this.saveSuccess ? $r('app.color.success_light') : $r('app.color.danger_light'))
            .borderRadius(10)
            .width('100%')
            .justifyContent(FlexAlign.Center)
          }

          if (this.hasDormitorySettings) {
            Button(this.isLoading ? '保存中...' : '保存设置')
              .width('100%')
              .height(50)
              .fontSize(16)
              .fontWeight(FontWeight.Medium)
              .backgroundColor($r('app.color.primary'))
              .fontColor($r('app.color.button_text'))
              .borderRadius(14)
              .enabled(!this.isLoading)
              .onClick(() => {
                this.saveSettings();
              })
          }

          Column({ space: 12 }) {
            Row() {
              Image($r('app.media.info_circle'))
                .width(16)
                .height(16)
                .fillColor($r('app.color.text_tertiary'))
              Text('温馨提示')
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor($r('app.color.text_secondary'))
                .margin({ left: 6 })
            }

            Column({ space: 6 }) {
              Row() {
                Text('•')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                Text('系统将每小时自动检查电费余额')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
              }
              .width('100%')
              .alignItems(VerticalAlign.Top)

              Row() {
                Text('•')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                Text('密码将加密存储，仅用于查询电费')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
              }
              .width('100%')
              .alignItems(VerticalAlign.Top)

              Row() {
                Text('•')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                Text('余额低于阈值时将收到推送通知提醒')
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
              }
              .width('100%')
              .alignItems(VerticalAlign.Top)
            }
          }
          .width('100%')
          .padding(16)
          .backgroundColor($r('app.color.card_bg'))
          .borderRadius(12)
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 24 })
      }
      .scrollBar(BarState.Off)
      .edgeEffect(EdgeEffect.Spring)
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.page_bg'))
    .padding({ top: 48 })
  }
}
