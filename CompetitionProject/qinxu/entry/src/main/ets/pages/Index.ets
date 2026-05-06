/**
 * 登录页面模块
 * 
 * 提供用户登录功能，包括：
 * - 账号密码登录
 * - 记住密码
 * - 自动登录
 * - 隐私协议确认
 * 
 * @module Index
 */

import { router } from '@kit.ArkUI';
import { promptAction } from '@kit.ArkUI';
import { preferences } from '@kit.ArkData';
import { ApiClient } from '../utils/ApiClient';
import { PrivacyDialog, hasAgreedPrivacy } from '../components/PrivacyDialog';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'login_preferences';
/** 用户名存储键名 */
const KEY_USERNAME = 'username';
/** 密码存储键名 */
const KEY_PASSWORD = 'password';
/** 记住密码存储键名 */
const KEY_REMEMBER_ME = 'rememberMe';
/** 自动登录存储键名 */
const KEY_AUTO_LOGIN = 'autoLogin';

/**
 * 登录页面组件
 * 
 * 提供用户登录界面，支持记住密码和自动登录功能
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/Index' });
 * ```
 */
@Entry
@Component
struct Login {
  /** 用户名 */
  @State username: string = '';
  /** 密码 */
  @State password: string = '';
  /** 记住密码标志 */
  @State rememberMe: boolean = false;
  /** 自动登录标志 */
  @State autoLogin: boolean = false;
  /** 显示隐私对话框标志 */
  @State showPrivacyDialog: boolean = false;
  /** 是否已同意隐私协议 */
  @State hasAgreedPrivacy: boolean = false;
  /** 是否正在初始化 */
  @State isInitializing: boolean = true;

  build() {
    Stack() {
      if (this.isInitializing) {
        Column() {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'));
        }
        .width('100%')
        .height('100%')
        .justifyContent(FlexAlign.Center)
        .backgroundColor($r('app.color.background_primary'));
      } else {
        Column() {
          Column() {
            Text('青序')
              .fontSize(32)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))
              .margin({ bottom: 8 })
            
            Text('Campus Assistant')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
              .margin({ bottom: 60 })
          }
          .width('100%')
          .alignItems(HorizontalAlign.Center)
          .margin({ top: 120 })

          Column({ space: 16 }) {
            TextInput({ placeholder: '请输入学号', text: this.username })
              .onChange((value: string) => { this.username = value; })
              .height(50)
              .borderRadius(12)
              .backgroundColor($r('app.color.background_secondary'))
              .padding({ left: 16, right: 16 })
              .placeholderColor($r('app.color.text_tertiary'))
              .fontSize(16)

            TextInput({ placeholder: '请输入密码', text: this.password })
              .onChange((value: string) => { this.password = value; })
              .type(InputType.Password)
              .height(50)
              .borderRadius(12)
              .backgroundColor($r('app.color.background_secondary'))
              .padding({ left: 16, right: 16 })
              .placeholderColor($r('app.color.text_tertiary'))
              .fontSize(16)

            Row({ space: 40 }) {
              Row() {
                Checkbox()
                  .select(this.rememberMe)
                  .onChange((value: boolean) => {
                    this.rememberMe = value;
                    if (!value) {
                      this.autoLogin = false;
                    }
                  })
                  .selectedColor($r('app.color.primary'))
                Text('记住密码')
                  .fontSize(14)
                  .fontColor($r('app.color.text_primary'))
                  .margin({ left: 8 })
              }

              Row() {
                Checkbox()
                  .select(this.autoLogin)
                  .onChange((value: boolean) => {
                    if (value && !this.rememberMe) {
                      this.rememberMe = true;
                    }
                    this.autoLogin = value;
                  })
                  .selectedColor($r('app.color.primary'))
                Text('自动登录')
                  .fontSize(14)
                  .fontColor($r('app.color.text_primary'))
                  .margin({ left: 8 })
              }
            }
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
            .margin({ top: 8 })

            Button('登录')
              .width('100%')
              .height(50)
              .fontSize(17)
              .fontWeight(FontWeight.Medium)
              .backgroundColor($r('app.color.primary'))
              .borderRadius(12)
              .margin({ top: 24 })
              .onClick(() => {
                this.handleLogin();
              })
          }
          .width('100%')
          .padding({ left: 32, right: 32 })
          .layoutWeight(1)

          Text('© 青序 v1.0')
            .fontSize(12)
            .fontColor($r('app.color.text_tertiary'))
            .margin({ bottom: 32 })
        }
        .width('100%')
        .height('100%')
        .backgroundColor($r('app.color.background_primary'))
      }

      if (this.showPrivacyDialog) {
        PrivacyDialog({
          isVisible: $showPrivacyDialog,
          onAgreed: () => {
            this.onPrivacyAgreed();
          }
        })
      }
    }
    .width('100%')
    .height('100%')
    .onAppear(() => {
      this.checkPrivacyAgreement();
    })
  }

  /**
   * 检查隐私协议状态
   * 如果未同意则显示隐私对话框
   */
  private async checkPrivacyAgreement(): Promise<void> {
    try {
      const agreed = await hasAgreedPrivacy();
      this.hasAgreedPrivacy = agreed;
      
      if (!agreed) {
        this.isInitializing = false;
        this.showPrivacyDialog = true;
      } else {
        this.loadSavedCredentials();
      }
    } catch (err) {
      console.error('检查隐私协议状态失败:', JSON.stringify(err));
      this.isInitializing = false;
      this.showPrivacyDialog = true;
    }
  }

  /**
   * 隐私协议同意回调
   * 加载保存的登录凭证
   */
  private onPrivacyAgreed(): void {
    this.hasAgreedPrivacy = true;
    this.loadSavedCredentials();
  }

  /**
   * 处理登录操作
   * 验证输入并调用登录API
   */
  private async handleLogin(): Promise<void> {
    if (!this.username.trim()) {
      promptAction.showToast({ message: '请输入学号' });
      return;
    }
    if (!this.password) {
      promptAction.showToast({ message: '请输入密码' });
      return;
    }

    promptAction.showToast({ message: '正在登录，请稍候…' });

    const result = await ApiClient.syncStudent(this.username, this.password);

    if (!result || !result.success) {
      promptAction.showToast({ message: result && result.message ? result.message : '登录失败，请稍后重试' });
      return;
    }

    if (this.rememberMe) {
      await this.saveCredentials();
    } else {
      await this.clearSavedCredentials();
    }

    AppStorage.SetOrCreate('studentId', this.username);

    promptAction.showToast({ message: '登录成功，正在同步数据…' });
    
    router.replaceUrl({ url: 'pages/MainPage' });
  }

  /**
   * 保存登录凭证
   * 将用户名、密码和设置存储到本地
   */
  private async saveCredentials(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      await dataPreferences.put(KEY_USERNAME, this.username);
      await dataPreferences.put(KEY_PASSWORD, this.password);
      await dataPreferences.put(KEY_REMEMBER_ME, this.rememberMe);
      await dataPreferences.put(KEY_AUTO_LOGIN, this.autoLogin);
      await dataPreferences.flush();
      console.info('保存登录凭证成功');
    } catch (err) {
      console.error('保存登录凭证失败:', JSON.stringify(err));
    }
  }

  /**
   * 清除保存的登录凭证
   */
  private async clearSavedCredentials(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      await dataPreferences.delete(KEY_USERNAME);
      await dataPreferences.delete(KEY_PASSWORD);
      await dataPreferences.delete(KEY_REMEMBER_ME);
      await dataPreferences.delete(KEY_AUTO_LOGIN);
      await dataPreferences.flush();
      console.info('清除保存的登录凭证成功');
    } catch (err) {
      console.error('清除保存的登录凭证失败:', JSON.stringify(err));
    }
  }

  /**
   * 加载保存的登录凭证
   * 如果开启自动登录则自动跳转
   */
  private async loadSavedCredentials(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      const savedUsername = await dataPreferences.get(KEY_USERNAME, '');
      const savedPassword = await dataPreferences.get(KEY_PASSWORD, '');
      const savedRememberMe = await dataPreferences.get(KEY_REMEMBER_ME, false);
      const savedAutoLogin = await dataPreferences.get(KEY_AUTO_LOGIN, false);

      this.username = String(savedUsername);
      this.password = String(savedPassword);
      this.rememberMe = Boolean(savedRememberMe);
      this.autoLogin = Boolean(savedAutoLogin);

      console.info('加载保存的登录凭证成功:', {
        username: this.username,
        rememberMe: this.rememberMe,
        autoLogin: this.autoLogin
      });

      if (this.autoLogin && this.username && this.password) {
        AppStorage.SetOrCreate('studentId', this.username);
        await router.replaceUrl({ url: 'pages/MainPage' });
        this.silentLogin();
      } else {
        this.isInitializing = false;
      }
    } catch (err) {
      console.error('加载保存的登录凭证失败:', JSON.stringify(err));
      this.isInitializing = false;
    }
  }

  /**
   * 静默登录
   * 使用保存的凭证自动登录
   */
  private async silentLogin(): Promise<void> {
    try {
      const result = await ApiClient.syncStudent(this.username, this.password);

      if (result && result.success) {
        promptAction.showToast({ message: '登录成功' });
      } else {
        promptAction.showToast({ message: '自动登录失败，请重新登录' });
        await router.replaceUrl({ url: 'pages/Index' });
      }
    } catch (err) {
      console.error('静默登录失败:', JSON.stringify(err));
      promptAction.showToast({ message: '自动登录失败，请重新登录' });
      await router.replaceUrl({ url: 'pages/Index' });
    }
  }
}
