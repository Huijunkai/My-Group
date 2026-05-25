/**
 * 账号设置页面模块
 * 
 * 提供账号管理功能，包括：
 * - 账号信息查看
 * - 密码管理
 * 
 * @module AccountSettingsPage
 */

import { router } from '@kit.ArkUI';
import { preferences } from '@kit.ArkData';
import { BusinessError } from '@kit.BasicServicesKit';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'login_preferences';
/** 用户名存储键名 */
const KEY_USERNAME = 'username';
/** 密码存储键名 */
const KEY_PASSWORD = 'password';

/**
 * 账号设置页面组件
 * 
 * 展示和管理用户账号信息
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/AccountSettingsPage' });
 * ```
 */
@Entry
@Component
struct AccountSettingsPage {
  /** 用户名 */
  @State username: string = '';
  /** 密码 */
  @State password: string = '';
  /** 密码是否可见 */
  @State passwordVisible: boolean = false;

  async aboutToAppear(): Promise<void> {
    this.username = await this.loadUsername();
    this.password = await this.loadPassword();
  }

  onBackPress(): boolean {
    router.back();
    return true;
  }

  async loadUsername(): Promise<string> {
    try {
      let dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      let value = await dataPreferences.get(KEY_USERNAME, '') as string;
      console.info('Succeeded in getting value of key.');
      return value;
    } catch (err) {
      let error = err as BusinessError;
      console.error('Failed to get value of key. Cause: ' + error.message);
      return '';
    }
  }

  async loadPassword(): Promise<string> {
    try {
      let dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      let value = await dataPreferences.get(KEY_PASSWORD, '') as string;
      console.info('Succeeded in getting value of key.');
      return value;
    } catch (err) {
      let error = err as BusinessError;
      console.error('Failed to get value of key. Cause: ' + error.message);
      return '';
    }
  }

  build() {
    Column() {
      Column()
        .width('100%')
        .height(44)
      this.buildHeader()
      this.buildContent()
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_quaternary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
  }

  @Builder
  buildHeader() {
    Row() {
      Image($r('app.media.chevron_left'))
        .width(24)
        .height(24)
        .fillColor($r('app.color.text_primary'))
        .onClick(() => {
          router.back();
        })
      Text('账号设置')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
        .textAlign(TextAlign.Center)
      Blank()
        .width(24)
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
  }

  @Builder
  buildContent() {
    Scroll() {
      Column({ space: 16 }) {
        this.buildAccount()
        this.buildDashedDivider()
        this.buildPassword()
      }
      .width('100%')
      .padding(16)
    }
    .layoutWeight(1)
  }

  @Builder
  buildAccount() {
    Column() {
      Row() {
        Text('学号')
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))
          .fontWeight(FontWeight.Medium)
          .layoutWeight(1)
        Text(this.username)
          .fontSize(15)
          .fontColor($r('app.color.text_tertiary'))
      }
      .width('100%')
      .height(52)
      .padding({ left: 16, right: 16 })
      .alignItems(VerticalAlign.Center)
    }
    .width('100%')
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildPassword() {
    Column() {
      Row() {
        Row({ space: 8 }) {
          Text('密码')
            .fontSize(15)
            .fontColor($r('app.color.text_primary'))
            .fontWeight(FontWeight.Medium)
          
          Text('(长按可查看)')
            .fontSize(10)
            .fontColor($r('app.color.text_secondary'))
        }

        if (this.passwordVisible) {
          Text(this.password)
            .fontSize(15)
            .fontColor($r('app.color.text_tertiary'))
            .layoutWeight(1)
            .textAlign(TextAlign.End)
        } else {
          Text('••••••••')
            .fontSize(15)
            .fontColor($r('app.color.text_tertiary'))
            .layoutWeight(1)
            .textAlign(TextAlign.End)
        }
      }
      .width('100%')
      .height(52)
      .padding({ left: 16, right: 16 })
      .alignItems(VerticalAlign.Center)
    }
    .width('100%')
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
    .gesture(
      LongPressGesture({ repeat: false })
        .onAction(() => {
          this.passwordVisible = true;
        })
        .onActionEnd(() => {
          this.passwordVisible = false;
        })
    )
  }

  @Builder
  buildDashedDivider() {
    Row() {
      ForEach([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], (index: number) => {
        if (index % 2 === 0) {
          Column()
            .width(8)
            .height(1)
            .backgroundColor($r('app.color.divider'))
        } else {
          Column()
            .width(4)
            .height(1)
        }
      })
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
  }
}
