/**
 * 个人中心页面模块
 * 
 * 提供用户信息和设置功能，包括：
 * - 个人信息展示
 * - 账号设置
 * - 推送设置
 * - 隐私设置
 * - 主题设置
 * - 退出登录
 * 
 * @module ProfilePage
 */

import { router, promptAction } from '@kit.ArkUI';
import { preferences } from '@kit.ArkData';
import { ApiClient } from '../utils/ApiClient';
import { DataStore } from '../utils/DataStore';
import { ThemeColors, DEFAULT_COLORS } from '../utils/ThemeColorManager';
import { common, ConfigurationConstant } from '@kit.AbilityKit';

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
 * 菜单项接口
 */
interface MenuItem {
  /** 菜单标题 */
  title: string;
  /** 点击回调 */
  action: () => void;
}

/**
 * 个人中心页面组件
 * 
 * 展示用户个人信息和设置入口
 * 
 * @example
 * ```typescript
 * // 在 MainPage 中通过 GlassMenubar 加载
 * ```
 */
@Entry
@Component
struct ProfilePage {
  /** 用户姓名 */
  @State userName: string = '';
  /** 性别 */
  @State gender: string = '';
  /** 学号 */
  @State studentId: string = '';
  /** 入学时间 */
  @State enrollmentTime: string = '';
  /** 学院 */
  @State college: string = '';
  /** 专业 */
  @State major: string = '';
  /** 班级 */
  @State className: string = '';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 滚动偏移量 */
  @State scrollOffset: number = 0;
  /** 状态栏透明度 */
  @State statusBarOpacity: number = 0;
  /** 暗色模式标志（全局存储链接） */
  @StorageLink('isDarkMode') isDarkMode: boolean = false;
  /** 主题颜色（全局存储链接） */
  @StorageLink('themeColors') themeColors: ThemeColors = DEFAULT_COLORS;
  /** 数据是否已加载 */
  private dataLoaded: boolean = false;
  /** 滚动控制器 */
  private scroller: Scroller = new Scroller();

  aboutToAppear() {
    this.initDarkMode();
    if (!this.dataLoaded) {
      this.loadStudentInfo();
      this.dataLoaded = true;
    }
  }

  /**
   * 初始化暗色模式
   */
  private initDarkMode(): void {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const isDark = context.config.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
      AppStorage.SetOrCreate('isDarkMode', isDark);
    } catch (e) {
      console.warn('ProfilePage: Failed to init dark mode:', JSON.stringify(e));
    }
  }

  /**
   * 加载学生信息
   */
  private async loadStudentInfo(): Promise<void> {
    const storedStudentId = AppStorage.Get('studentId') as string;
    if (!storedStudentId) {
      console.log('ProfilePage: No studentId found');
      this.isLoading = false;
      return;
    }
    this.studentId = storedStudentId;
    console.log('ProfilePage: Loading info for studentId:', this.studentId);

    await this.loadCachedInfo();
    this.isLoading = false;
  }

  /**
   * 加载缓存的学生信息
   */
  private async loadCachedInfo(): Promise<void> {
    try {
      const dataStore = DataStore.getInstance();
      const cachedInfo = await dataStore.loadStudentInfo();
      
      if (cachedInfo) {
        console.log('ProfilePage: Found cached info:', cachedInfo);
        if (cachedInfo.name) this.userName = cachedInfo.name;
        if (cachedInfo.gender) this.gender = cachedInfo.gender;
        if (cachedInfo.enrollmentYear) this.enrollmentTime = cachedInfo.enrollmentYear;
        if (cachedInfo.college) this.college = cachedInfo.college;
        if (cachedInfo.major) this.major = cachedInfo.major;
        if (cachedInfo.className) this.className = cachedInfo.className;
      }

      const cachedName = AppStorage.Get('userName') as string;
      if (cachedName && cachedName !== '' && !this.userName) {
        this.userName = cachedName;
      }
    } catch (error) {
      console.error('ProfilePage: Failed to load cached info:', error);
    }
  }

  /**
   * 解析并更新学生信息
   * 
   * @param rawInfo - 原始信息对象
   */
  private async parseAndUpdateInfo(rawInfo: Object): Promise<void> {
    console.log('ProfilePage: Raw info type:', typeof rawInfo);
    console.log('ProfilePage: Raw info:', JSON.stringify(rawInfo));
    
    let info: Record<string, Object> = {};
    
    if (rawInfo && typeof rawInfo === 'object') {
      const jsonStr = JSON.stringify(rawInfo);
      info = JSON.parse(jsonStr) as Record<string, Object>;
    }
    
    console.log('ProfilePage: Parsed info:', JSON.stringify(info));
    
    const keys = ['name', 'gender', 'enrollmentYear', 'college', 'major', 'className'];
    for (const key of keys) {
      if (info[key] !== undefined && info[key] !== null) {
        const value = String(info[key]);
        if (value !== '' && value !== 'null' && value !== 'undefined') {
          switch (key) {
            case 'name':
              this.userName = value;
              break;
            case 'gender':
              this.gender = value;
              break;
            case 'enrollmentYear':
              this.enrollmentTime = value;
              break;
            case 'college':
              this.college = value;
              break;
            case 'major':
              this.major = value;
              break;
            case 'className':
              this.className = value;
              break;
          }
        }
      }
    }
    
    console.log('ProfilePage: Final loaded data:', {
      userName: this.userName,
      gender: this.gender,
      enrollmentTime: this.enrollmentTime,
      college: this.college,
      major: this.major,
      className: this.className
    });
    
    this.cacheInfo();
  }

  /**
   * 缓存学生信息到本地存储
   */
  private cacheInfo(): void {
    AppStorage.SetOrCreate('userName', this.userName);
    AppStorage.SetOrCreate('gender', this.gender);
    AppStorage.SetOrCreate('enrollmentTime', this.enrollmentTime);
    AppStorage.SetOrCreate('college', this.college);
    AppStorage.SetOrCreate('major', this.major);
    AppStorage.SetOrCreate('className', this.className);

    const dataStore = DataStore.getInstance();
    dataStore.saveStudentInfo({
      studentId: this.studentId,
      name: this.userName,
      gender: this.gender,
      enrollmentYear: this.enrollmentTime,
      className: this.className,
      major: this.major,
      college: this.college
    });
  }

  build() {
    Stack() {
      Column()
        .width('100%')
        .height('100%')
        .backgroundColor($r('app.color.background_tertiary'))

      Scroll(this.scroller) {
        Column({ space: 12 }) {
          Column()
            .width('100%')
            .height(44)

          this.buildHeader()
          this.buildPersonalInfoCard()
          this.buildDashedDivider()
          this.buildMenuSection('常用功能', [
            { title: '账号设置', action: (): void => this.goToAccountSettings() },
            { title: '推送设置', action: (): void => this.goToPushSettings() },
            { title: '隐私设置', action: (): void => this.goToPrivacySettings() },
            { title: '主题设置', action: (): void => this.goToThemeSettings() }
          ])
          this.buildMenuSection('其他', [
            { title: '关于我们', action: (): void => this.goToAbout() }
          ])
          
          Button('退出登录')
            .width('90%')
            .height(44)
            .fontSize(15)
            .fontColor('#E84026')
            .backgroundColor('rgba(232, 64, 38, 0.1)')
            .borderRadius(22)
            .margin({ top: 8, bottom: 16 })
            .onClick((): void => this.handleLogout())
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 0, bottom: 80 })
        .justifyContent(FlexAlign.Start)
      }
      .scrollBar(BarState.Off)
      .edgeEffect(EdgeEffect.Spring)
      .width('100%')
      .height('100%')
      .onScroll(() => {
        this.handleScroll();
      })

      Column() {
        Column()
          .width('100%')
          .height(100)
          .linearGradient({
            angle: 180,
            colors: this.isDarkMode ? [
              [`rgba(0, 0, 0, ${0.98 * this.statusBarOpacity})`, 0],
              [`rgba(0, 0, 0, ${0.85 * this.statusBarOpacity})`, 0.3],
              [`rgba(0, 0, 0, ${0.5 * this.statusBarOpacity})`, 0.6],
              [`rgba(0, 0, 0, ${0.1 * this.statusBarOpacity})`, 0.85],
              [`rgba(0, 0, 0, 0)`, 1]
            ] : [
              [`rgba(245, 245, 247, ${0.98 * this.statusBarOpacity})`, 0],
              [`rgba(245, 245, 247, ${0.85 * this.statusBarOpacity})`, 0.3],
              [`rgba(245, 245, 247, ${0.5 * this.statusBarOpacity})`, 0.6],
              [`rgba(245, 245, 247, ${0.1 * this.statusBarOpacity})`, 0.85],
              [`rgba(245, 245, 247, 0)`, 1]
            ]
          })
      }
      .width('100%')
      .position({ x: 0, y: 0 })
    }
    .width('100%')
    .height('100%')
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  /**
   * 处理滚动事件
   * 更新状态栏透明度
   */
  private handleScroll(): void {
    const offset = this.scroller.currentOffset().yOffset;
    this.scrollOffset = offset;
    const maxOffset = 100;
    this.statusBarOpacity = Math.min(1, Math.max(0, offset / maxOffset));
  }

  /**
   * 构建页面头部
   */
  @Builder
  buildHeader() {
    Column() {
      Row() {
        Text('我的')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
      }
      .width('100%')
    }
    .width('100%')
    .padding(20)
    .borderRadius(16)
    .linearGradient({
      angle: 135,
      colors: [[this.themeColors.profileGradientStart, 0], [this.themeColors.profileGradientEnd, 1]]
    })
  }

  /**
   * 构建个人信息卡片
   */
  @Builder
  buildPersonalInfoCard() {
    Column() {
      Row() {
        Stack() {
          Text(this.userName.charAt(0) || '未')
            .fontSize(32)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))
        }
        .width(64)
        .height(64)
        .backgroundColor(this.themeColors.avatarBackground)
        .borderRadius(32)

        Column({ space: 6 }) {
          Row() {
            Text(this.userName || '未获取')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
            Text(`（${this.gender || '未知'}）`)
              .fontSize(13)
              .fontColor($r('app.color.text_secondary'))
              .margin({ left: 6 })
          }
          .width('100%')

          Text(`学号：${this.studentId || '未获取'}`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')

          Text(`${this.enrollmentTime || '未获取'}入学 · ${this.college || '未获取'}`)
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')

          Text(`${this.major || '未获取'} · ${this.className || '未获取'}`)
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')
        }
        .margin({ left: 12 })
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')
      .alignItems(VerticalAlign.Top)
    }
    .width('100%')
    .padding(20)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildInfoTag(label: string, value: string) {
    Column({ space: 2 }) {
      Text(label)
        .fontSize(11)
        .fontColor($r('app.color.text_secondary'))
      
      Text(value)
        .fontSize(12)
        .fontColor($r('app.color.text_primary'))
        .fontWeight(FontWeight.Medium)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .width('100%')
    }
    .width('100%')
    .padding({ top: 6, bottom: 6, left: 8, right: 8 })
    .backgroundColor($r('app.color.background_tertiary'))
    .borderRadius(8)
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  buildInfoRow(label: string, value: string) {
    Row() {
      Text(label)
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .width(72)
        .flexShrink(0)

      Text(value)
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
  }

  @Builder
  buildMenuSection(title: string, items: Array<MenuItem>) {
    Column() {
      Text(title)
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')
        .padding({ left: 4, bottom: 8, top: 4 })

      Column() {
        ForEach(items, (item: MenuItem, index: number) => {
          this.buildMenuItem(item, index === items.length - 1)
        })
      }
      .width('100%')
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(16)
    }
    .width('100%')
  }

  @Builder
  buildMenuItem(item: MenuItem, isLast: boolean) {
    Row() {
      Text(item.title)
        .fontSize(15)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)

      Image($r('app.media.chevron_right'))
        .width(14)
        .height(14)
        .fillColor($r('app.color.text_tertiary'))
    }
    .width('100%')
    .height(52)
    .padding({ left: 16, right: 16 })
    .onClick((): void => item.action())
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

  /**
   * 处理退出登录
   */
  private handleLogout(): void {
    interface DialogButton {
      text: string;
      color: Resource | string;
    }

    const buttons: DialogButton[] = [
      { text: '取消', color: $r('app.color.text_primary') },
      { text: '确定', color: $r('app.color.primary') }
    ];

    promptAction.showDialog({
      title: '确认退出',
      message: '您确定要退出登录吗？',
      buttons: buttons
    }).then((result: promptAction.ShowDialogSuccessResponse) => {
      if (result.index === 1) {
        this.clearLoginState();
        router.replaceUrl({ url: 'pages/Index' });
        promptAction.showToast({ message: '已退出登录' });
      }
    });
  }

  /**
   * 清除登录状态
   */
  private clearLoginState(): void {
    try {
      AppStorage.SetOrCreate('studentId', '');
      AppStorage.SetOrCreate('userName', '');
      AppStorage.SetOrCreate('gender', '');
      AppStorage.SetOrCreate('enrollmentTime', '');
      AppStorage.SetOrCreate('college', '');
      AppStorage.SetOrCreate('major', '');
      AppStorage.SetOrCreate('className', '');
      
      preferences.getPreferences(getContext(), PREFERENCES_NAME).then(async (dataPreferences: preferences.Preferences) => {
        const rememberMe = await dataPreferences.get(KEY_REMEMBER_ME, false) as boolean;
        
        if (rememberMe) {
          dataPreferences.delete(KEY_AUTO_LOGIN);
        } else {
          dataPreferences.delete(KEY_USERNAME);
          dataPreferences.delete(KEY_PASSWORD);
          dataPreferences.delete(KEY_REMEMBER_ME);
          dataPreferences.delete(KEY_AUTO_LOGIN);
        }
        
        dataPreferences.flush();
      });
    } catch (err) {
      console.error('清除登录状态失败:', JSON.stringify(err));
    }
  }

  private showToast(message: string): void {
    promptAction.showToast({ message: `${message}功能开发中` });
  }

  private goToPrivacySettings(): void {
    router.pushUrl({ url: 'pages/PrivacySettingsPage' });
  }

  private goToAbout(): void {
    router.pushUrl({ url: 'pages/AboutPage' });
  }

  private goToAccountSettings(): void {
    router.pushUrl({ url: 'pages/AccountSettingsPage' });
  }

  private goToThemeSettings(): void {
    router.pushUrl({ url: 'pages/ThemeSettingsPage' });
  }

  private goToPushSettings(): void {
    router.pushUrl({ url: 'pages/PushSettingsPage' });
  }
}

export { ProfilePage };
