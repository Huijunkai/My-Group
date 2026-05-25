/**
 * 首页模块
 * 
 * 提供应用主页功能，包括：
 * - 今日/明日课程展示
 * - 天气信息展示
 * - 公告通知
 * - 宿舍信息卡片
 * - 快捷功能入口
 * 
 * @module HomePage
 */

import { router, window, display } from '@kit.ArkUI';
import { ApiClient, Announcement, AnnouncementDetail, AnnouncementAttachment } from '../utils/ApiClient';
import { DataStore, ScheduleData } from '../utils/DataStore';
import { SettingsData } from '../model/SettingsData';
import { ThemeColors, DEFAULT_COLORS } from '../utils/ThemeColorManager';
import { NotificationManager, NotificationStatus } from '../utils/NotificationManager';
import { WeatherService, WeatherData, MinutelyData } from '../utils/WeatherService';
import { common, abilityAccessCtrl, Permissions, ConfigurationConstant, Want } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { notificationManager } from '@kit.NotificationKit';
import { DormitoryCard } from '../components/DormitoryCard';

/**
 * 课程信息接口
 */
interface CourseInfo {
  /** 课程名称 */
  name: string;
  /** 时间范围 */
  time: string;
  /** 上课地点 */
  location: string;
  /** 授课教师 */
  teacher: string;
  /** 节次 */
  period: string;
  /** 开始时间 */
  startTime: string;
  /** 结束时间 */
  endTime: string;
  /** 是否已结束 */
  isEnded: boolean;
  /** 是否正在进行 */
  isCurrent: boolean;
  /** 是否即将上课 */
  isUpcomingSoon: boolean;
  /** 是否即将下课 */
  isEndingSoon: boolean;
  /** 课程状态 */
  status: string;
  /** 是否有冲突 */
  hasConflict?: boolean;
  /** 冲突课程列表 */
  conflictWith?: string[];
}

/**
 * 后端课程数据接口
 */
interface BackendCourse {
  /** 星期几 */
  dayOfWeek?: string | number;
  /** 周次字符串 */
  weeks?: string;
  /** 周次 */
  week?: number;
  /** 节次 */
  period?: string;
  /** 课程名称 */
  name?: string;
  /** 上课地点 */
  location?: string;
  /** 授课教师 */
  teacher?: string;
}

/**
 * 时间段配置接口
 */
interface TimeSlotConfig {
  /** 节次 */
  period: string;
  /** 开始时间 */
  startTime: string;
  /** 结束时间 */
  endTime: string;
}

/**
 * 提醒配置接口
 */
interface GeneratedObjectLiteralInterface_1 {
  /** 即将上课阈值（分钟） */
  upcomingSoonThreshold: number;
  /** 即将下课阈值（分钟） */
  endingSoonThreshold: number;
}

/**
 * 首页组件
 * 
 * 展示今日课程、天气、公告等信息
 * 
 * @example
 * ```typescript
 * // 在 MainPage 中通过 GlassMenubar 加载
 * ```
 */
@Entry
@Component
struct HomePage {
  @State currentDate: string = this.getCurrentDate();
  @State greeting: string = this.getGreeting();
  @State greetingSub: string = this.getGreetingSub();
  @State userName: string = '同学';
  @State studentId: string = '';
  @State todayCourses: CourseInfo[] = [];
  @State tomorrowCourses: CourseInfo[] = [];
  @State isLoading: boolean = true;
  @State settings: SettingsData = new SettingsData();
  @State scrollOffset: number = 0;
  @State statusBarOpacity: number = 0;
  @State weatherData: WeatherData | null = null;
  @State weatherLoading: boolean = true;
  @State isTablet: boolean = false;
  @State screenWidth: number = 0;
  @StorageLink('isDarkMode') isDarkMode: boolean = false;
  @StorageLink('themeColors') themeColors: ThemeColors = DEFAULT_COLORS;
  @StorageLink('notificationStatus') notificationStatus: NotificationStatus = {
    hasNewGrade: false,
    hasNewCredit: false,
    hasNewExam: false,
    lastGradeUpdate: '',
    lastCreditUpdate: '',
    lastExamUpdate: ''
  };
  @State announcements: Announcement[] = [];
  @State announcementLoading: boolean = true;
  @State selectedAnnouncement: Announcement | null = null;
  @State announcementDetail: AnnouncementDetail | null = null;
  @State announcementDetailLoading: boolean = false;
  @StorageLink('showAnnouncementDetail') @Watch('onAnnouncementDetailChanged') showAnnouncementDetail: boolean = false;
  @StorageLink('announcementPushUrl') @Watch('onAnnouncementPushReceived') announcementPushUrl: string = '';
  @State showAnnouncementList: boolean = false;
  @State announcementScrollIndex: number = 0;
  @State announcementTranslateY: number = 0;
  @State currentPage: number = 1;
  @State pageSize: number = 10;
  @State totalAnnouncements: number = 0;
  private announcementScrollTimer: number | null = null;
  private dataLoaded: boolean = false;
  private updateTimer: number | null = null;
  private weatherScrollTimer: number | null = null;
  private notificationManager: NotificationManager = NotificationManager.getInstance();
  private weatherService: WeatherService = WeatherService.getInstance();
  private scroller: Scroller = new Scroller();
  private weatherScroller: Scroller = new Scroller();
  private weatherContentWidth: number = 0;
  private weatherScrollX: number = 0;

  private timeSlots: TimeSlotConfig[] = [
    { period: '01-02节', startTime: '08:30', endTime: '10:05' },
    { period: '03-04节', startTime: '10:25', endTime: '12:00' },
    { period: '05-06节', startTime: '14:30', endTime: '16:05' },
    { period: '07-08节', startTime: '16:15', endTime: '17:50' },
    { period: '09-10节', startTime: '18:20', endTime: '19:55' },
    { period: '11-12节', startTime: '20:05', endTime: '21:40' },
    { period: '01节', startTime: '08:30', endTime: '09:15' },
    { period: '02节', startTime: '09:20', endTime: '10:05' },
    { period: '03节', startTime: '10:25', endTime: '11:05' },
    { period: '04节', startTime: '11:15', endTime: '12:00' },
    { period: '05节', startTime: '14:30', endTime: '15:15' },
    { period: '06节', startTime: '15:20', endTime: '16:05' },
    { period: '07节', startTime: '16:15', endTime: '17:00' },
    { period: '08节', startTime: '17:05', endTime: '17:50' },
    { period: '09节', startTime: '18:20', endTime: '19:05' },
    { period: '10节', startTime: '19:10', endTime: '19:55' },
    { period: '11节', startTime: '20:05', endTime: '20:50' },
    { period: '12节', startTime: '20:55', endTime: '21:40' }
  ];

  aboutToAppear() {
    this.initDarkMode();
    this.detectDeviceType();
    if (!this.dataLoaded) {
      this.loadSettings();
      this.dataLoaded = true;
    }
    this.startRealTimeUpdates();
    setTimeout(() => {
      this.requestLocationPermissionIfNeeded();
      this.loadWeatherData();
      this.loadAnnouncements();
    }, 500);
    this.startWeatherAutoScroll();
    this.startAnnouncementAutoScroll();
    this.listenForPushMessages();
  }

  private detectDeviceType(): void {
    try {
      const displayInfo = display.getDefaultDisplaySync();
      this.screenWidth = displayInfo.width / displayInfo.densityPixels;
      this.isTablet = this.screenWidth >= 600;
      console.info(`HomePage: Screen width: ${this.screenWidth}, isTablet: ${this.isTablet}`);
    } catch (error) {
      console.error('HomePage: Failed to detect device type:', JSON.stringify(error));
      this.isTablet = false;
    }
  }

  private listenForPushMessages(): void {
    AppStorage.setOrCreate('announcementPushUrl', '');
    AppStorage.setOrCreate('announcementPushTitle', '');
  }

  private initDarkMode(): void {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const isDark = context.config.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
      AppStorage.SetOrCreate('isDarkMode', isDark);
    } catch (e) {
      console.warn('HomePage: Failed to init dark mode:', JSON.stringify(e));
    }
  }

  aboutToDisappear() {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
    if (this.weatherScrollTimer) {
      clearInterval(this.weatherScrollTimer);
      this.weatherScrollTimer = null;
    }
    if (this.announcementScrollTimer) {
      clearInterval(this.announcementScrollTimer);
      this.announcementScrollTimer = null;
    }
  }

  private startWeatherAutoScroll(): void {
    if (this.isTablet) {
      return;
    }
    this.weatherScrollTimer = setInterval(() => {
      if (this.weatherData && this.weatherContentWidth > 0) {
        this.weatherScrollX += 0.5;
        if (this.weatherScrollX >= this.weatherContentWidth) {
          this.weatherScrollX = 0;
          this.weatherScroller.scrollTo({ xOffset: 0, yOffset: 0, animation: false });
        } else {
          this.weatherScroller.scrollBy(0.5, 0);
        }
      }
    }, 16);
  }

  private startAnnouncementAutoScroll(): void {
    this.announcementScrollTimer = setInterval(() => {
      if (this.announcements.length > 0 && !this.showAnnouncementList) {
        this.announcementTranslateY = -36;
        setTimeout(() => {
          this.announcementScrollIndex = (this.announcementScrollIndex + 1) % this.announcements.length;
          this.announcementTranslateY = 0;
        }, 300);
      }
    }, 3000);
  }

  private async loadAnnouncements(): Promise<void> {
    try {
      this.announcementLoading = true;
      const result = await ApiClient.getAnnouncements(this.pageSize, (this.currentPage - 1) * this.pageSize);
      this.announcements = result.announcements;
      this.totalAnnouncements = result.total;
      this.announcementScrollIndex = 0;
      console.info('HomePage: Announcements loaded:', this.announcements.length, 'total:', this.totalAnnouncements);
    } catch (error) {
      console.error('HomePage: Failed to load announcements:', JSON.stringify(error));
    } finally {
      this.announcementLoading = false;
    }
  }

  private async openAnnouncementDetail(announcement: Announcement): Promise<void> {
    this.selectedAnnouncement = announcement;
    AppStorage.SetOrCreate('showAnnouncementDetail', true);
    this.announcementDetailLoading = true;
    this.announcementDetail = null;

    try {
      const detail = await ApiClient.getAnnouncementDetail(announcement.url);
      if (detail) {
        this.announcementDetail = detail;
      }
    } catch (error) {
      console.error('HomePage: Failed to load announcement detail:', JSON.stringify(error));
    } finally {
      this.announcementDetailLoading = false;
    }
  }

  private onAnnouncementDetailChanged(): void {
    if (!this.showAnnouncementDetail) {
      this.selectedAnnouncement = null;
      this.announcementDetail = null;
    }
  }

  private onAnnouncementPushReceived(): void {
    if (this.announcementPushUrl && this.announcementPushUrl.length > 0) {
      console.info('HomePage: Received announcement push, URL:', this.announcementPushUrl);
      const title = AppStorage.Get('announcementPushTitle') as string || '公告详情';
      this.selectedAnnouncement = {
        title: title,
        url: this.announcementPushUrl,
        date: '',
        id: 0
      };
      this.openAnnouncementDetail(this.selectedAnnouncement);
      AppStorage.SetOrCreate('announcementPushUrl', '');
      AppStorage.SetOrCreate('announcementPushTitle', '');
    }
  }

  private closeAnnouncementDetail(): void {
    AppStorage.SetOrCreate('showAnnouncementDetail', false);
    this.selectedAnnouncement = null;
    this.announcementDetail = null;
  }

  private openAttachment(url: string): void {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const want: Want = {
        action: 'ohos.want.action.viewData',
        entities: ['entity.system.browsable'],
        uri: url
      };
      context.startAbility(want);
    } catch (error) {
      console.error('HomePage: Failed to open attachment:', JSON.stringify(error));
    }
  }

  private startRealTimeUpdates(): void {
    this.updateTimer = setInterval(() => {
      this.updateCourseStatuses();
      this.updateTimeInfo();
    }, 60000);
  }

  private async loadWeatherData(): Promise<void> {
    try {
      this.weatherLoading = true;
      this.weatherContentWidth = 0;
      this.weatherScrollX = 0;
      this.weatherScroller.scrollTo({ xOffset: 0, yOffset: 0, animation: false });
      const weather = await this.weatherService.getWeather();
      if (weather) {
        this.weatherData = weather;
        console.info('HomePage: Weather data loaded successfully');
      }
    } catch (error) {
      console.error('HomePage: Failed to load weather data:', JSON.stringify(error));
    } finally {
      this.weatherLoading = false;
    }
  }

  private async requestLocationPermissionIfNeeded(): Promise<void> {
    const KEY_PERMISSIONS_REQUESTED = 'permissionsRequested';
    const PREFERENCES_NAME = 'app_preferences';
    
    try {
      const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      const alreadyRequested = await dataPreferences.get(KEY_PERMISSIONS_REQUESTED, false);
      
      if (alreadyRequested) {
        console.info('HomePage: 权限已请求过，跳过');
        return;
      }

      const context = getContext() as common.UIAbilityContext;
      const atManager = abilityAccessCtrl.createAtManager();
      
      const permissionsToRequest: Permissions[] = [
        'ohos.permission.APPROXIMATELY_LOCATION',
        'ohos.permission.LOCATION',
        'ohos.permission.PUBLISH_AGENT_REMINDER'
      ];
      
      const permissionsNeeded: Permissions[] = [];
      
      for (const permission of permissionsToRequest) {
        const status = await atManager.checkAccessToken(
          context.applicationInfo.accessTokenId,
          permission
        );
        
        if (status !== abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) {
          permissionsNeeded.push(permission);
        }
      }
      
      if (permissionsNeeded.length > 0) {
        try {
          await atManager.requestPermissionsFromUser(context, permissionsNeeded);
          console.info('HomePage: 系统权限请求完成');
        } catch (e) {
          console.warn('HomePage: 系统权限请求失败:', JSON.stringify(e));
        }
      }

      try {
        const notificationEnabled = await notificationManager.isNotificationEnabled();
        if (!notificationEnabled) {
          await notificationManager.requestEnableNotification();
          console.info('HomePage: 通知权限请求完成');
        }
      } catch (e) {
        console.warn('HomePage: 通知权限请求失败:', JSON.stringify(e));
      }

      await dataPreferences.put(KEY_PERMISSIONS_REQUESTED, true);
      await dataPreferences.flush();
      console.info('HomePage: 权限请求流程完成');
    } catch (err) {
      console.error('HomePage: 请求权限失败:', JSON.stringify(err));
    }
  }

  private updateTimeInfo(): void {
    this.currentDate = this.getCurrentDate();
    this.greeting = this.getGreeting();
    this.greetingSub = this.getGreetingSub();
  }

  private updateCourseStatuses(): void {
    this.todayCourses = this.todayCourses.map(course => {
      const courseStatus = this.getCourseStatus(course.startTime, course.endTime);
      const updatedCourse: CourseInfo = {
        name: course.name,
        time: course.time,
        location: course.location,
        teacher: course.teacher,
        period: course.period,
        startTime: course.startTime,
        endTime: course.endTime,
        isEnded: courseStatus === 'ended',
        isCurrent: courseStatus === 'current',
        isUpcomingSoon: courseStatus === 'upcoming_soon',
        isEndingSoon: courseStatus === 'ending_soon',
        status: courseStatus,
        hasConflict: course.hasConflict,
        conflictWith: course.conflictWith
      };
      return updatedCourse;
    });
  }

  private async loadSettings(): Promise<void> {
    try {
      const dataStore = DataStore.getInstance();
      const loadedSettings: SettingsData | null = await dataStore.loadSettings();
      if (loadedSettings) {
        this.settings = loadedSettings;
      }

      const cachedInfo = await dataStore.loadStudentInfo();
      if (cachedInfo && cachedInfo.name) {
        this.userName = cachedInfo.name;
        console.log('HomePage: Loaded cached user name:', this.userName);
      }
    } catch (error) {
      console.error('HomePage: Failed to load settings:', error);
    }
    setTimeout(() => {
      this.loadData();
    }, 100);
  }

  private async loadData(): Promise<void> {
    const storedStudentId = AppStorage.Get('studentId') as string;
    if (!storedStudentId) {
      this.isLoading = false;
      return;
    }
    this.studentId = storedStudentId;

    try {
      const dataStore = DataStore.getInstance();
      
      const cachedInfo = await dataStore.loadStudentInfo();
      if (cachedInfo && cachedInfo.name) {
        this.userName = cachedInfo.name;
        console.log('HomePage: Loaded cached user name:', this.userName);
      }

      const scheduleData = await dataStore.loadScheduleData();
      if (scheduleData && scheduleData.courses) {
        const courses = scheduleData.courses;
        const today = new Date().getDay() || 7;
        const tomorrow = today === 7 ? 1 : today + 1;
        const currentWeek = this.getCurrentWeek();
        let tomorrowWeek: number = currentWeek;
        if (today === 7) {
          tomorrowWeek = currentWeek + 1;
        }

        const dayMap: Record<string, number> = {
          '星期一': 1,
          '星期二': 2,
          '星期三': 3,
          '星期四': 4,
          '星期五': 5,
          '星期六': 6,
          '星期日': 7
        };

        const todayCourses: CourseInfo[] = courses.filter((c: BackendCourse) => {
          const dayOfWeekStr = String(c.dayOfWeek || '');
          const dayOfWeek = dayMap[dayOfWeekStr] || 1;
          const week = Number(c.week || 1);
          return dayOfWeek === today && week === currentWeek;
        }).map((c: BackendCourse) => {
          const period = String(c.period || '');
          const timeConfig = this.getTimeConfig(period);
          const courseStatus = this.getCourseStatus(timeConfig.startTime, timeConfig.endTime);
          
          const courseInfo: CourseInfo = {
            name: String(c.name || ''),
            time: period,
            location: String(c.location || ''),
            teacher: String(c.teacher || ''),
            period: period,
            startTime: timeConfig.startTime,
            endTime: timeConfig.endTime,
            isEnded: courseStatus === 'ended',
            isCurrent: courseStatus === 'current',
            isUpcomingSoon: courseStatus === 'upcoming_soon',
            isEndingSoon: courseStatus === 'ending_soon',
            status: courseStatus,
            hasConflict: false,
            conflictWith: []
          };
          return courseInfo;
        }).sort((a: CourseInfo, b: CourseInfo) => {
          return a.startTime.localeCompare(b.startTime);
        });

        const tomorrowCourses: CourseInfo[] = courses.filter((c: BackendCourse) => {
          const dayOfWeekStr = String(c.dayOfWeek || '');
          const dayOfWeek = dayMap[dayOfWeekStr] || 1;
          const week = Number(c.week || 1);
          return dayOfWeek === tomorrow && week === tomorrowWeek;
        }).map((c: BackendCourse) => {
          const period = String(c.period || '');
          const timeConfig = this.getTimeConfig(period);
          
          const courseInfo: CourseInfo = {
            name: String(c.name || ''),
            time: period,
            location: String(c.location || ''),
            teacher: String(c.teacher || ''),
            period: period,
            startTime: timeConfig.startTime,
            endTime: timeConfig.endTime,
            isEnded: false,
            isCurrent: false,
            isUpcomingSoon: false,
            isEndingSoon: false,
            status: 'upcoming',
            hasConflict: false,
            conflictWith: []
          };
          return courseInfo;
        }).sort((a: CourseInfo, b: CourseInfo) => {
          return a.startTime.localeCompare(b.startTime);
        });

        this.detectCourseConflicts(todayCourses);
        this.detectCourseConflicts(tomorrowCourses);

        this.todayCourses = this.mergeSameCourses(todayCourses);
        this.tomorrowCourses = this.mergeSameCourses(tomorrowCourses);
        console.log('HomePage: Loaded today courses:', this.todayCourses.length);
        console.log('HomePage: Loaded tomorrow courses:', this.tomorrowCourses.length);
      }
    } catch (error) {
      console.error('HomePage: Failed to load data:', error);
    }
    this.isLoading = false;
  }

  private getTimeConfig(period: string): TimeSlotConfig {
    // 首先检查是否是跨多个标准连堂课的格式，如 "09-12节"
    const match = period.match(/(\d{1,2})[\-~](\d{1,2})节/);
    if (match) {
      const start = parseInt(match[1]);
      const end = parseInt(match[2]);
      
      // 查找开始节和结束节的时间配置
      const startSlot = this.timeSlots.find(s => s.period === `${String(start).padStart(2, '0')}节`);
      const endSlot = this.timeSlots.find(s => s.period === `${String(end).padStart(2, '0')}节`);
      
      if (startSlot && endSlot) {
        return { period: period, startTime: startSlot.startTime, endTime: endSlot.endTime };
      }
    }
    
    // 处理特殊格式，如 "09-10-11-12节"
    const multiMatch = period.match(/(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})节/);
    if (multiMatch) {
      const start = parseInt(multiMatch[1]);
      const end = parseInt(multiMatch[4]);
      
      // 查找开始节和结束节的时间配置
      const startSlot = this.timeSlots.find(s => s.period === `${String(start).padStart(2, '0')}节`);
      const endSlot = this.timeSlots.find(s => s.period === `${String(end).padStart(2, '0')}节`);
      
      if (startSlot && endSlot) {
        return { period: period, startTime: startSlot.startTime, endTime: endSlot.endTime };
      }
    }
    
    // 然后检查是否是标准的连堂课格式
    for (const slot of this.timeSlots) {
      if (period.includes(slot.period) || slot.period.includes(period)) {
        return slot;
      }
    }
    
    // 处理单节课格式
    const singleMatch = period.match(/(\d{1,2})节/);
    if (singleMatch) {
      const periodNum = parseInt(singleMatch[1]);
      const slot = this.timeSlots.find(s => s.period === `${String(periodNum).padStart(2, '0')}节`);
      if (slot) {
        return slot;
      }
    }
    
    // 处理特殊格式，如 "01,02节"
    const specialMatch = period.match(/(\d{1,2}),\d{1,2}节/);
    if (specialMatch) {
      const start = parseInt(specialMatch[1]);
      const startSlot = this.timeSlots.find(s => s.period === `${String(start).padStart(2, '0')}节`);
      if (startSlot) {
        // 尝试找到结束节的时间配置
        const endMatch = period.match(/\d{1,2},(\d{1,2})节/);
        if (endMatch) {
          const end = parseInt(endMatch[1]);
          const endSlot = this.timeSlots.find(s => s.period === `${String(end).padStart(2, '0')}节`);
          if (endSlot) {
            return { period: period, startTime: startSlot.startTime, endTime: endSlot.endTime };
          }
        }
        return startSlot;
      }
    }
    
    // 默认返回一个合理的时间范围
    return { period: period, startTime: '08:30', endTime: '10:05' };
  }

  // 课程提醒时间配置
  private reminderConfig: GeneratedObjectLiteralInterface_1 = {
    upcomingSoonThreshold: 15, // 即将上课阈值（分钟）
    endingSoonThreshold: 5,   // 即将下课阈值（分钟）
  };

  private getCourseStatus(startTime: string, endTime: string): string {
    const now = new Date();
    const currentTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    
    // 将时间转换为分钟数，便于计算时间差
    const timeToMinutes = (time: string): number => {
      const timeParts = time.split(':').map(Number);
      const hours = timeParts[0];
      const minutes = timeParts[1];
      return hours * 60 + minutes;
    };
    
    const currentMinutes = timeToMinutes(currentTime);
    const startMinutes = timeToMinutes(startTime);
    const endMinutes = timeToMinutes(endTime);
    
    // 计算时间差
    const timeUntilStart = startMinutes - currentMinutes;
    const timeUntilEnd = endMinutes - currentMinutes;
    
    if (timeUntilStart > this.reminderConfig.upcomingSoonThreshold) {
      return 'upcoming';
    } else if (timeUntilStart <= this.reminderConfig.upcomingSoonThreshold && timeUntilStart > 0) {
      return 'upcoming_soon'; // 即将上课
    } else if (timeUntilEnd > this.reminderConfig.endingSoonThreshold) {
      return 'current';
    } else if (timeUntilEnd <= this.reminderConfig.endingSoonThreshold && timeUntilEnd >= 0) {
      return 'ending_soon'; // 即将下课
    } else {
      return 'ended';
    }
  }

  private getCurrentWeek(): number {
    if (!this.settings.semesterStartDate) {
      return 1;
    }
    
    const today = new Date();
    const startDate = new Date(this.settings.semesterStartDate);
    
    if (isNaN(startDate.getTime())) {
      return 1;
    }
    
    const todayOnlyDate = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const startOnlyDate = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate());
    
    const timeDiff = todayOnlyDate.getTime() - startOnlyDate.getTime();
    const dayDiff = Math.ceil(timeDiff / (1000 * 3600 * 24));
    
    const currentWeek = Math.max(1, Math.ceil((dayDiff + 1) / 7));
    
    return Math.min(currentWeek, this.settings.totalWeeks);
  }

  private getCurrentDate(): string {
    const now = new Date();
    const month = now.getMonth() + 1;
    const day = now.getDate();
    const weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
    return `${month}月${day}日 ${weekdays[now.getDay()]}`;
  }

  private getGreeting(): string {
      const hour = new Date().getHours();
      // 5:00-6:00 破晓时分
      if (hour >= 5 && hour < 6) {
        return '东方欲白';
      }
      // 6:00-7:00 清晨晨光
      else if (hour >= 6 && hour < 7) {
        return '晓色初分';
      }
      // 7:00-8:00 晨光渐盛
      else if (hour >= 7 && hour < 8) {
        return '晨光熹微';
      }
      // 8:00-9:00 早间明朗
      else if (hour >= 8 && hour < 9) {
        return '晨光大亮';
      }
      // 9:00-10:00 上午初段
      else if (hour >= 9 && hour < 10) {
        return '日上三竿';
      }
      // 10:00-11:00 上午中段
      else if (hour >= 10 && hour < 11) {
        return '日升中天';
      }
      // 11:00-12:00 近午时分
      else if (hour >= 11 && hour < 12) {
        return '日近晌午';
      }
      // 12:00-13:00 正午时刻
      else if (hour >= 12 && hour < 13) {
        return '午时初刻';
      }
      // 13:00-14:00 午后初段
      else if (hour >= 13 && hour < 14) {
        return '午后方醒';
      }
      // 14:00-15:00 未时初段
      else if (hour >= 14 && hour < 15) {
        return '日昳初至';
      }
      // 15:00-16:00 未时中段
      else if (hour >= 15 && hour < 16) {
        return '日昃之离';
      }
      // 16:00-17:00 申时初段
      else if (hour >= 16 && hour < 17) {
        return '晡时初临';
      }
      // 17:00-18:00 申时中段
      else if (hour >= 17 && hour < 18) {
        return '夕阳西斜';
      }
      // 18:00-19:00 黄昏时分
      else if (hour >= 18 && hour < 19) {
        return '日暮黄昏';
      }
      // 19:00-20:00 入夜时分
      else if (hour >= 19 && hour < 20) {
        return '薄暮已至';
      }
      // 20:00-21:00 夜晚初段
      else if (hour >= 20 && hour < 21) {
        return '月升星明';
      }
      // 21:00-22:00 夜晚中段
      else if (hour >= 21 && hour < 22) {
        return '月升日落';
      }
      // 其他时段（非5-22点）兜底
      else {
        return '夜深人静';
      }
    }

  private getGreetingSub(): string {
      const hour = new Date().getHours();
      // 5:00-6:00 清晨破晓
      if (hour >= 5 && hour < 6) {
        return '东方欲晓，莫道君行早';
      }
      // 6:00-7:00 晨光初现
      else if (hour >= 6 && hour < 7) {
        return '晨兴理荒秽，带月荷锄归';
      }
      // 7:00-8:00 清晨盛景
      else if (hour >= 7 && hour < 8) {
        return '春眠不觉晓，处处闻啼鸟';
      }
      // 8:00-9:00 早间奋进
      else if (hour >= 8 && hour < 9) {
        return '一日之计在于晨';
      }
      // 9:00-10:00 上午治学
      else if (hour >= 9 && hour < 10) {
        return '学而不思则罔';
      }
      // 10:00-11:00 上午勤勉
      else if (hour >= 10 && hour < 11) {
        return '少壮不努力，老大徒伤悲';
      }
      // 11:00-12:00 近午时光
      else if (hour >= 11 && hour < 12) {
        return '日高烟敛，更看今日晴未';
      }
      // 12:00-13:00 正午休憩
      else if (hour >= 12 && hour < 13) {
        return '偷得浮生半日闲';
      }
      // 13:00-14:00 午后闲适
      else if (hour >= 13 && hour < 14) {
        return '午窗初睡起，幽兴付茶瓯';
      }
      // 14:00-15:00 午后慵懒
      else if (hour >= 14 && hour < 15) {
        return '美人帐中午睡起，钗横鬓亸慵添妆';
      }
      // 15:00-16:00 午后观景
      else if (hour >= 15 && hour < 16) {
        return '漠漠水田飞白鹭，阴阴夏木啭黄鹂';
      }
      // 16:00-17:00 晡时悠然
      else if (hour >= 16 && hour < 17) {
        return '日昳方炊饭，秋深始采茶';
      }
      // 17:00-18:00 夕阳初现
      else if (hour >= 17 && hour < 18) {
        return '夕阳无限好';
      }
      // 18:00-19:00 黄昏时分
      else if (hour >= 18 && hour < 19) {
        return '大漠孤烟直，长河落日圆';
      }
      // 19:00-20:00 傍晚闲逸
      else if (hour >= 19 && hour < 20) {
        return '晚来天欲雪，能饮一杯无';
      }
      // 20:00-21:00 入夜温馨
      else if (hour >= 20 && hour < 21) {
        return '天阶夜色凉如水，坐看牵牛织女星';
      }
      // 21:00-22:00 夜阑静心
      else if (hour >= 21 && hour < 22) {
        return '明月几时有';
      }
      // 其他时段（非5-22点）兜底
      else {
        return '夜深风竹敲秋韵';
      }
    }

  private getCurrentCourse(): CourseInfo | null {
    return this.todayCourses.find(c => c.isCurrent) || null;
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
      }, (index: number) => `divider_${index}`)
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
    .margin({ top: 4, bottom: 4 })
  }

  build() {
    Stack() {
      Column()
        .width('100%')
        .height('100%')
        .backgroundColor($r('app.color.background_quaternary'))

      Scroll(this.scroller) {
        Column({ space: 8 }) {
          Column()
            .width('100%')
            .height(44)

          this.buildHeader()
          this.buildWeatherCard()
          this.buildAnnouncementCard()
          this.buildScheduleCardsRow()
          DormitoryCard()
          this.buildQuickActionsCard()
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

      this.buildAnnouncementDetailModal()
    }
    .width('100%')
    .height('100%')
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  private handleScroll(): void {
    const offset = this.scroller.currentOffset().yOffset;
    this.scrollOffset = offset;
    const maxOffset = 100;
    this.statusBarOpacity = Math.min(1, Math.max(0, offset / maxOffset));
  }

  @Builder
  buildHeader() {
    Column() {
      Row() {
        Column({ space: 4 }) {
          Text(this.currentDate)
            .fontSize(13)
            .fontColor($r('app.color.text_primary'))

          Row({ space: 6 }) {
            Text(`${this.greeting}，`)
              .fontSize(22)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))

            Text(this.userName)
              .fontSize(22)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .layoutWeight(1)
          }

          Text(this.greetingSub)
            .fontSize(13)
            .fontColor($r('app.color.text_primary'))
            .margin({ top: 4 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')
    }
    .width('100%')
    .padding(20)
    .borderRadius(16)
    .linearGradient({
      angle: 135,
      colors: [[this.themeColors.headerGradientStart, 0], [this.themeColors.headerGradientEnd, 1]]
    })
  }

  @Builder
  buildWeatherCard() {
    Column() {
      if (this.weatherLoading) {
        Row() {
          LoadingProgress()
            .width(16)
            .height(16)
            .color($r('app.color.text_secondary'))
          Text('加载天气中...')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .margin({ left: 6 })
        }
        .width('100%')
        .justifyContent(FlexAlign.Center)
        .padding({ top: 8, bottom: 8 })
      } else if (this.weatherData) {
        if (this.isTablet) {
          Row({ space: 16 }) {
            this.buildWeatherTabletItem('位置', this.weatherData.location)
            this.buildWeatherScrollDivider()
            this.buildWeatherTabletItem('天气', this.weatherData.condition)
            this.buildWeatherScrollDivider()
            if (this.weatherData.tempMin !== 0 || this.weatherData.tempMax !== 0) {
              this.buildWeatherTabletItem('气温', `${Math.round(this.weatherData.tempMin)}°~${Math.round(this.weatherData.tempMax)}°`)
              this.buildWeatherScrollDivider()
            }
            this.buildWeatherTabletItem('当前', `${Math.round(this.weatherData.temperature)}°`)
            this.buildWeatherScrollDivider()
            this.buildWeatherTabletItem('降雨', `${this.weatherData.precipProbability}%`)
            this.buildWeatherScrollDivider()
            this.buildWeatherTabletItem('湿度', `${this.weatherData.humidity}%`)
            this.buildWeatherScrollDivider()
            this.buildWeatherTabletItem('风向', `${this.weatherData.windDirection} ${Math.round(this.weatherData.windSpeed)}m/s`)
            this.buildWeatherScrollDivider()
            this.buildWeatherTabletItem('体感', `${Math.round(this.weatherData.feelsLike)}°`)
          }
          .width('100%')
          .justifyContent(FlexAlign.SpaceEvenly)
        } else {
          Scroll(this.weatherScroller) {
            Row({ space: 0 }) {
              this.buildWeatherScrollItem('位置', this.weatherData.location)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('天气', this.weatherData.condition)
              this.buildWeatherScrollDivider()
              if (this.weatherData.tempMin !== 0 || this.weatherData.tempMax !== 0) {
                this.buildWeatherScrollItem('气温', `${Math.round(this.weatherData.tempMin)}°~${Math.round(this.weatherData.tempMax)}°`)
                this.buildWeatherScrollDivider()
              }
              this.buildWeatherScrollItem('当前', `${Math.round(this.weatherData.temperature)}°`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('降雨', `${this.weatherData.precipProbability}%`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('湿度', `${this.weatherData.humidity}%`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('风向', `${this.weatherData.windDirection} ${Math.round(this.weatherData.windSpeed)}m/s`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('体感', `${Math.round(this.weatherData.feelsLike)}°`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('位置', this.weatherData.location)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('天气', this.weatherData.condition)
              this.buildWeatherScrollDivider()
              if (this.weatherData.tempMin !== 0 || this.weatherData.tempMax !== 0) {
                this.buildWeatherScrollItem('气温', `${Math.round(this.weatherData.tempMin)}°~${Math.round(this.weatherData.tempMax)}°`)
                this.buildWeatherScrollDivider()
              }
              this.buildWeatherScrollItem('当前', `${Math.round(this.weatherData.temperature)}°`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('降雨', `${this.weatherData.precipProbability}%`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('湿度', `${this.weatherData.humidity}%`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('风向', `${this.weatherData.windDirection} ${Math.round(this.weatherData.windSpeed)}m/s`)
              this.buildWeatherScrollDivider()
              this.buildWeatherScrollItem('体感', `${Math.round(this.weatherData.feelsLike)}°`)
            }
            .onAreaChange((oldValue: Area, newValue: Area) => {
              const width = newValue.width as number;
              if (width > 0 && this.weatherContentWidth === 0) {
                this.weatherContentWidth = width / 2;
              }
            })
          }
          .scrollable(ScrollDirection.Horizontal)
          .scrollBar(BarState.Off)
          .width('100%')
          .height(36)
        }
      }
    }
    .width('100%')
    .padding({ left: 12, right: 12, top: 6, bottom: 6 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(12)
    .onClick(() => {
      this.weatherService.clearCache();
      this.loadWeatherData();
    })
  }

  @Builder
  buildWeatherTabletItem(label: string, value: string) {
    Row({ space: 6 }) {
      Text(label)
        .fontSize(13)
        .fontColor($r('app.color.text_tertiary'))
      Text(value)
        .fontSize(14)
        .fontColor($r('app.color.text_primary'))
        .fontWeight(FontWeight.Medium)
    }
    .padding({ left: 12, right: 12 })
    .flexShrink(0)
  }

  @Builder
  buildWeatherScrollItem(label: string, value: string) {
    Row({ space: 4 }) {
      Text(label)
        .fontSize(11)
        .fontColor($r('app.color.text_tertiary'))
      Text(value)
        .fontSize(12)
        .fontColor($r('app.color.text_primary'))
        .fontWeight(FontWeight.Medium)
    }
    .padding({ left: 8, right: 8 })
    .flexShrink(0)
  }

  @Builder
  buildWeatherScrollDivider() {
    Column()
      .width(1)
      .height(14)
      .backgroundColor($r('app.color.divider'))
      .flexShrink(0)
  }

  @Builder
  buildAnnouncementCard() {
    Column() {
      Row() {
        Image($r('app.media.bell_fill'))
          .width(16)
          .height(16)
          .fillColor($r('app.color.primary'))
        
        Text('公告')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .margin({ left: 6 })
        
        Blank()
        
        if (this.announcements.length > 0) {
          Image($r('app.media.chevron_right'))
            .width(14)
            .height(14)
            .fillColor($r('app.color.text_secondary'))
            .rotate({ angle: this.showAnnouncementList ? 90 : 0 })
            .animation({ duration: 200 })
        }
      }
      .width('100%')
      .margin({ bottom: 8 })

      if (this.announcementLoading) {
        Row() {
          LoadingProgress()
            .width(16)
            .height(16)
            .color($r('app.color.text_secondary'))
          Text('加载公告中...')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .margin({ left: 6 })
        }
        .width('100%')
        .justifyContent(FlexAlign.Center)
        .padding({ top: 8, bottom: 8 })
      } else if (this.announcements.length === 0) {
        Text('暂无公告')
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')
          .textAlign(TextAlign.Center)
          .padding({ top: 8, bottom: 8 })
      } else if (this.showAnnouncementList) {
        Column({ space: 0 }) {
          ForEach(this.announcements, (announcement: Announcement, index: number) => {
            Row() {
              if (announcement.date) {
                Text(announcement.date)
                  .fontSize(11)
                  .fontColor($r('app.color.text_tertiary'))
                  .width(70)
                  .flexShrink(0)
              }
              Text(announcement.title)
                .fontSize(13)
                .fontColor($r('app.color.text_primary'))
                .fontWeight(FontWeight.Medium)
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
                .margin({ left: 8 })
                .layoutWeight(1)
            }
            .width('100%')
            .height(32)
            .onClick(() => {
              this.openAnnouncementDetail(announcement);
            })
          }, (announcement: Announcement, index: number) => `ann_list_${index}`)
          
          Row() {
            if (this.currentPage > 1) {
              Button('上一页')
                .fontSize(12)
                .fontColor($r('app.color.primary'))
                .backgroundColor('transparent')
                .margin({ right: 12 })
                .onClick(() => {
                  this.currentPage--;
                  this.loadAnnouncements();
                })
            }
            Text(`第 ${this.currentPage} 页`)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
            if (this.currentPage * this.pageSize < this.totalAnnouncements) {
              Button('下一页')
                .fontSize(12)
                .fontColor($r('app.color.primary'))
                .backgroundColor('transparent')
                .margin({ left: 12 })
                .onClick(() => {
                  this.currentPage++;
                  this.loadAnnouncements();
                })
            }
          }
          .width('100%')
          .justifyContent(FlexAlign.Center)
          .padding({ top: 12 })
        }
        .width('100%')
      } else {
        Stack() {
          ForEach(this.announcements, (announcement: Announcement, index: number) => {
            Row() {
              if (announcement.date) {
                Text(announcement.date)
                  .fontSize(11)
                  .fontColor($r('app.color.text_tertiary'))
                  .width(70)
                  .flexShrink(0)
              }
              Text(announcement.title)
                .fontSize(13)
                .fontColor($r('app.color.text_primary'))
                .fontWeight(FontWeight.Medium)
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
                .margin({ left: 8 })
                .layoutWeight(1)
            }
            .width('100%')
            .height(36)
            .onClick(() => {
              this.openAnnouncementDetail(announcement);
            })
            .position({ x: 0, y: index === this.announcementScrollIndex ? 0 : (index < this.announcementScrollIndex ? -36 : 36) })
            .opacity(index === this.announcementScrollIndex ? 1 : 0)
            .animation({
              duration: 400,
              curve: Curve.EaseInOut
            })
          }, (announcement: Announcement, index: number) => `ann_scroll_${index}`)
        }
        .width('100%')
        .height(36)
        .clip(true)
      }
    }
    .width('100%')
    .padding({ left: 12, right: 12, top: 10, bottom: 10 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(12)
    .onClick(() => {
      if (this.announcements.length > 0) {
        this.showAnnouncementList = !this.showAnnouncementList;
      }
    })
  }

  @Builder
  buildAnnouncementDetailModal() {
    if (this.showAnnouncementDetail) {
      Column() {
        Column() {
          Row() {
            Image($r('app.media.xmark'))
              .width(20)
              .height(20)
              .fillColor($r('app.color.text_primary'))
              .onClick(() => {
                this.closeAnnouncementDetail();
              })
            
            Text('公告详情')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))
              .layoutWeight(1)
              .textAlign(TextAlign.Center)
            
            Column()
              .width(20)
              .height(20)
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 16, bottom: 12 })

          if (this.announcementDetailLoading) {
            Column() {
              LoadingProgress()
                .width(32)
                .height(32)
                .color($r('app.color.primary'))
              Text('加载中...')
                .fontSize(14)
                .fontColor($r('app.color.text_secondary'))
                .margin({ top: 12 })
            }
            .width('100%')
            .height(300)
            .justifyContent(FlexAlign.Center)
          } else if (this.announcementDetail) {
            Scroll() {
              Column({ space: 12 }) {
                Text(this.announcementDetail.title)
                  .fontSize(16)
                  .fontWeight(FontWeight.Bold)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .textAlign(TextAlign.Center)

                if (this.announcementDetail.date) {
                  Text(this.announcementDetail.date)
                    .fontSize(12)
                    .fontColor($r('app.color.text_secondary'))
                }

                Divider()
                  .color($r('app.color.divider'))
                  .margin({ top: 4, bottom: 4 })

                Text(this.formatAnnouncementContent(this.announcementDetail.content))
                  .fontSize(14)
                  .fontColor($r('app.color.text_primary'))
                  .lineHeight(26)
                  .width('100%')

                if (this.announcementDetail.attachments && this.announcementDetail.attachments.length > 0) {
                  Divider()
                    .color($r('app.color.divider'))
                    .margin({ top: 8, bottom: 4 })

                  Text('附件')
                    .fontSize(14)
                    .fontWeight(FontWeight.Medium)
                    .fontColor($r('app.color.text_primary'))
                    .width('100%')

                  ForEach(this.announcementDetail.attachments, (attachment: AnnouncementAttachment) => {
                    Row() {
                      Image($r('app.media.doc_plaintext_and_pencil_fill'))
                        .width(18)
                        .height(18)

                      Text(attachment.name)
                        .fontSize(13)
                        .fontColor($r('app.color.primary'))
                        .maxLines(2)
                        .textOverflow({ overflow: TextOverflow.Ellipsis })
                        .margin({ left: 8 })
                        .layoutWeight(1)
                    }
                    .width('100%')
                    .padding({ top: 8, bottom: 8 })
                    .onClick(() => {
                      this.openAttachment(attachment.url);
                    })
                  }, (attachment: AnnouncementAttachment) => attachment.name)
                }
              }
              .width('100%')
              .padding(16)
            }
            .scrollBar(BarState.Auto)
            .width('100%')
            .layoutWeight(1)
          } else {
            Column() {
              Text('加载失败')
                .fontSize(14)
                .fontColor($r('app.color.text_secondary'))
            }
            .width('100%')
            .height(200)
            .justifyContent(FlexAlign.Center)
          }
        }
        .width('90%')
        .constraintSize({ maxHeight: '80%' })
        .backgroundColor($r('app.color.background_primary'))
        .borderRadius(16)
      }
      .width('100%')
      .height('100%')
      .backgroundColor('rgba(0, 0, 0, 0.5)')
      .justifyContent(FlexAlign.Center)
      .onClick(() => {
        this.closeAnnouncementDetail();
      })
      .gesture(
        PanGesture({ direction: PanDirection.Horizontal })
          .onActionEnd((event: GestureEvent) => {
            if (Math.abs(event.offsetX) > 80) {
              this.closeAnnouncementDetail();
            }
          })
      )
    }
  }

  private formatAnnouncementContent(html: string): string {
    let content = html
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<\/p>/gi, '\n\n')
      .replace(/<\/div>/gi, '\n')
      .replace(/<li[^>]*>/gi, '• ')
      .replace(/<\/li>/gi, '\n')
      .replace(/<a[^>]*href="[^"]*\.(pdf|doc|docx|xls|xlsx|zip|rar|7z|ppt|pptx|txt|wps)"[^>]*>[^<]*<\/a>/gi, '')
      .replace(/<[^>]+>/g, '')
      .replace(/&nbsp;/g, ' ')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/首页\s*>\s*通知公告/gi, '')
      .replace(/来源[：:]\s*/gi, '')
      .replace(/作者[：:]\s*[^\n]*/gi, '')
      .replace(/编辑[：:]\s*[^\n]*/gi, '')
      .replace(/浏览[：:]\s*[^\n]*/gi, '')
      .replace(/发布时间[：:]\s*[^\n]*/gi, '')
      .replace(/打印/g, '')
      .replace(/\d+\.\s*附件\d*[：:]?[^\n]*\.(pdf|doc|docx|xls|xlsx|zip|rar|7z|ppt|pptx|txt|wps)/gi, '')
      .replace(/附件\d*[：:][^\n]*/gi, '')
      .replace(/^[一二三四五六七八九十]+\s*[、.．]\s*/gm, '')
      .replace(/^\d+\s*[、.．]\s*/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^[ \t]+\n/gm, '\n')
      .replace(/\n[ \t]+$/gm, '\n')
      .replace(/^\s+/, '')
      .replace(/\s+$/, '');

    return content;
  }

  @Builder
  buildCurrentCourseCard() {
    if (this.getCurrentCourse()) {
      Column() {
        Row() {
          Column({ space: 4 }) {
            Row({ space: 8 }) {
              Text('正在进行')
                .fontSize(12)
                .fontColor('#FFFFFF')
                .backgroundColor($r('app.color.success'))
                .borderRadius(4)
                .padding({ left: 8, right: 8, top: 2, bottom: 2 })

              Text(`${this.getCurrentCourse()!.startTime}-${this.getCurrentCourse()!.endTime}`)
                .fontSize(12)
                .fontColor('rgba(255,255,255,0.8)')
            }

            Text(this.getCurrentCourse()!.name)
              .fontSize(20)
              .fontWeight(FontWeight.Bold)
              .fontColor('#FFFFFF')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)

          Column() {
            Image($r('app.media.local_fill'))
              .width(20)
              .height(20)
              .fillColor('#FFFFFF')
            Text(this.getCurrentCourse()!.location || '待定')
              .fontSize(12)
              .fontColor('rgba(255,255,255,0.9)')
              .margin({ top: 4 })
          }
          .alignItems(HorizontalAlign.Center)
        }
        .width('100%')
      }
      .width('100%')
      .padding(16)
      .linearGradient({
        angle: 135,
        colors: [[$r('app.color.success'), 0], ['#30D158', 1]]
      })
      .borderRadius(20)
    }
  }

  @Builder
  buildScheduleCardsRow() {
    Row({ space: 12 }) {
      Column() {
        this.buildTodayScheduleCardCompact()
      }
      .layoutWeight(1)

      Column() {
        this.buildTomorrowScheduleCardCompact()
      }
      .layoutWeight(1)
    }
    .width('100%')
    .margin({ top: 8, bottom: 8 })
  }

  @Builder
  buildTodayScheduleCardCompact() {
    Column({ space: 6 }) {
      Text('今日课程')
        .fontSize(15)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      if (this.todayCourses.length === 0) {
        Column({ space: 4 }) {
          Image($r('app.media.book_pages_fill_1'))
            .width(32)
            .height(32)
            .fillColor($r('app.color.text_tertiary'))

          Text('今日无课')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))

          if (this.tomorrowCourses.length > 0) {
            Text(`明日有 ${this.tomorrowCourses.length} 节课`)
              .fontSize(11)
              .fontColor($r('app.color.text_tertiary'))
              .margin({ top: 4 })
          }
        }
        .width('100%')
        .height(140)
        .justifyContent(FlexAlign.Center)
      } else if (this.getUpcomingCourses().length === 0) {
        Column({ space: 4 }) {
          Image($r('app.media.book_pages_fill_1'))
            .width(32)
            .height(32)
            .fillColor($r('app.color.text_tertiary'))

          Text('今日课程已全部结束')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))

          Text(`${this.todayCourses.length} 节课已完成 ✓`)
            .fontSize(11)
            .fontColor('#52c41a')
            .margin({ top: 4 })
        }
        .width('100%')
        .height(140)
        .justifyContent(FlexAlign.Center)
      } else {
        Scroll() {
          Column({ space: 8 }) {
            ForEach(this.getUpcomingCourses().slice(0, 3), (course: CourseInfo, index: number) => {
              Column({ space: 4 }) {
                Row() {
                  Text(course.startTime)
                    .fontSize(11)
                    .fontColor($r('app.color.primary'))
                    .fontWeight(FontWeight.Medium)

                  Text(course.period)
                    .fontSize(10)
                    .fontColor($r('app.color.text_secondary'))
                    .margin({ left: 6 })

                  Blank()

                  if (course.isCurrent) {
                    Text('进行中')
                      .fontSize(10)
                      .fontColor('#FFFFFF')
                      .padding({ left: 6, right: 6, top: 2, bottom: 2 })
                      .backgroundColor('#52c41a')
                      .borderRadius(8)
                  } else if (course.isEndingSoon) {
                    Text('即将下课')
                      .fontSize(10)
                      .fontColor('#fa8c16')
                      .fontWeight(FontWeight.Medium)
                  } else if (!course.isEnded && course.isUpcomingSoon) {
                    Text('即将开始')
                      .fontSize(10)
                      .fontColor($r('app.color.primary'))
                      .fontWeight(FontWeight.Medium)
                  }
                }
                .width('100%')

                Text(course.name)
                  .fontSize(13)
                  .fontColor($r('app.color.text_primary'))
                  .fontWeight(FontWeight.Medium)
                  .width('100%')
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })

                if (course.location) {
                  Text(course.location)
                    .fontSize(11)
                    .fontColor($r('app.color.text_tertiary'))
                    .width('100%')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                }
              }
              .width('100%')
              .padding(10)
              .backgroundColor($r('app.color.background_tertiary'))
              .borderRadius(8)
            }, (course: CourseInfo, index: number) => `today_compact_${index}`)

            if (this.getUpcomingCourses().length > 3) {
              Text(`还有 ${this.getUpcomingCourses().length - 3} 节课...`)
                .fontSize(11)
                .fontColor($r('app.color.text_secondary'))
                .width('100%')
                .textAlign(TextAlign.Center)
                .padding({ top: 4 })
            }
          }
          .width('100%')
        }
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)
        .width('100%')
        .height(140)
      }
    }
    .width('100%')
    .padding(14)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildTomorrowScheduleCardCompact() {
    Column({ space: 6 }) {
      Text('明日课程')
        .fontSize(15)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      if (this.tomorrowCourses.length === 0) {
        Column({ space: 4 }) {
          Image($r('app.media.book_pages_fill_1'))
            .width(32)
            .height(32)
            .fillColor($r('app.color.text_tertiary'))

          Text('明日无课')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .padding({ top: 8, bottom: 8 })
        .justifyContent(FlexAlign.Center)
      } else {
        Scroll() {
          Column({ space: 8 }) {
            ForEach(this.tomorrowCourses.slice(0, 3), (course: CourseInfo, index: number) => {
              Column({ space: 4 }) {
                Row() {
                  Text(course.startTime)
                    .fontSize(11)
                    .fontColor($r('app.color.primary'))
                    .fontWeight(FontWeight.Medium)

                  Text(course.period)
                    .fontSize(10)
                    .fontColor($r('app.color.text_secondary'))
                    .margin({ left: 6 })

                  Blank()

                  Text(`${course.period.replace(/[^0-9]/g, '').length > 2 ? '连堂' : ''}`)
                    .fontSize(10)
                    .fontColor($r('app.color.text_tertiary'))
                }
                .width('100%')

                Text(course.name)
                  .fontSize(13)
                  .fontColor($r('app.color.text_primary'))
                  .fontWeight(FontWeight.Medium)
                  .width('100%')
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })

                if (course.location) {
                  Text(course.location)
                    .fontSize(11)
                    .fontColor($r('app.color.text_tertiary'))
                    .width('100%')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                }
              }
              .width('100%')
              .padding(10)
              .backgroundColor($r('app.color.background_tertiary'))
              .borderRadius(8)
            }, (course: CourseInfo, index: number) => `tomorrow_compact_${index}`)

            if (this.tomorrowCourses.length > 3) {
              Text(`共 ${this.tomorrowCourses.length} 节课`)
                .fontSize(11)
                .fontColor($r('app.color.text_secondary'))
                .width('100%')
                .textAlign(TextAlign.Center)
                .padding({ top: 4 })
            }
          }
          .width('100%')
        }
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)
        .width('100%')
        .height(140)
      }
    }
    .width('100%')
    .padding(14)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildTodayScheduleCard() {
    Column({ space: 8 }) {
      Column() {
        Row() {
          Column({ space: 4 }) {
            Text('今日课程o(*￣︶￣*)o')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))

            if (this.getUpcomingCourses().length > 0) {
              Text(`噢>-< 今日还有 ${this.getUpcomingCourses().length} 节课<(￣︶￣)↗[GO!]`)
                .fontSize(12)
                .fontColor(this.themeColors.courseCountText)
            }
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
        }
        .width('100%')
        .margin({ bottom: 8 })

        if (this.todayCourses.length === 0) {
          Column({ space: 8 }) {
            Image($r('app.media.book_pages_fill_1'))
              .width(48)
              .height(48)
              .fillColor($r('app.color.text_tertiary'))

            Text('今日无课，请合理安排时间(￣o￣) . z Z')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .padding({ top: 8, bottom: 16 })
        } else if (this.getUpcomingCourses().length === 0) {
          Column({ space: 8 }) {
            Image($r('app.media.book_pages_fill_1'))
              .width(48)
              .height(48)
              .fillColor($r('app.color.text_tertiary'))

            Text('今日课程上完了༼ つ ◕_◕ ༽つ')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .padding({ top: 8, bottom: 16 })
        } else {
          Scroll() {
            Column({ space: 12 }) {
              ForEach(this.getUpcomingCourses(), (course: CourseInfo, index: number) => {
                this.buildCourseItem(course, index)
                if (index < this.getUpcomingCourses().length - 1) {
                  this.buildDashedDivider()
                }
              }, (course: CourseInfo, index: number) => `upcoming_${index}`)
            }
            .width('100%')
          }
          .scrollBar(BarState.Off)
          .edgeEffect(EdgeEffect.Spring)
          .width('100%')
          .height(150)
        }
      }
      .width('100%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(16)
    }
  }

  @Builder
  buildTomorrowScheduleCard() {
    Column() {
      Row() {
        Column({ space: 4 }) {
          Text('明日课程(*^▽^*)')
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))

          if (this.tomorrowCourses.length > 0) {
            Text(`明日有 ${this.tomorrowCourses.length} 节课，不要忘记吖(●'◡'●)！`)
              .fontSize(12)
              .fontColor(this.themeColors.courseCountText)
          }
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')
      .margin({ bottom: 8 })

      if (this.tomorrowCourses.length === 0) {
        Column({ space: 8 }) {
          Image($r('app.media.book_pages_fill_1'))
            .width(48)
            .height(48)
            .fillColor($r('app.color.text_tertiary'))

          Text('嚯...明日没课哟ヾ(≧▽≦*)o')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .padding({ top: 8, bottom: 16 })
      } else {
          Scroll() {
            Column({ space: 12 }) {
              ForEach(this.tomorrowCourses, (course: CourseInfo, index: number) => {
                this.buildCourseItem(course, index)
                if (index < this.tomorrowCourses.length - 1) {
                  this.buildDashedDivider()
                }
              }, (course: CourseInfo, index: number) => `tomorrow_${index}`)
            }
            .width('100%')
          }
          .scrollBar(BarState.Off)
          .edgeEffect(EdgeEffect.Spring)
          .width('100%')
          .height(150)
        }
    }
    .width('100%')
    .padding(20)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  private getUpcomingCourses(): CourseInfo[] {
    return this.todayCourses.filter(c => !c.isEnded);
  }

  private detectCourseConflicts(courses: CourseInfo[]): void {
    for (let i = 0; i < courses.length; i++) {
      courses[i].hasConflict = false;
      courses[i].conflictWith = [];
    }

    for (let i = 0; i < courses.length; i++) {
      for (let j = i + 1; j < courses.length; j++) {
        if (this.isSameCourse(courses[i], courses[j])) {
          continue;
        }
        if (this.isTimeOverlap(courses[i], courses[j])) {
          courses[i].hasConflict = true;
          courses[j].hasConflict = true;
          if (!courses[i].conflictWith!.includes(courses[j].name)) {
            courses[i].conflictWith!.push(courses[j].name);
          }
          if (!courses[j].conflictWith!.includes(courses[i].name)) {
            courses[j].conflictWith!.push(courses[i].name);
          }
        }
      }
    }
  }

  private isSameCourse(c1: CourseInfo, c2: CourseInfo): boolean {
    if (c1.name !== c2.name) {
      return false;
    }
    if (c1.location !== c2.location) {
      return false;
    }
    if (c1.teacher !== c2.teacher) {
      return false;
    }
    return true;
  }

  private mergeSameCourses(courses: CourseInfo[]): CourseInfo[] {
    if (courses.length === 0) {
      return courses;
    }

    const mergedCourses: CourseInfo[] = [];
    const usedIndices: Set<number> = new Set();

    for (let i = 0; i < courses.length; i++) {
      if (usedIndices.has(i)) {
        continue;
      }

      const currentCourse = courses[i];
      const sameCourseIndices: number[] = [i];

      for (let j = i + 1; j < courses.length; j++) {
        if (usedIndices.has(j)) {
          continue;
        }
        if (this.isSameCourse(currentCourse, courses[j])) {
          sameCourseIndices.push(j);
        }
      }

      if (sameCourseIndices.length === 1) {
        mergedCourses.push(currentCourse);
        usedIndices.add(i);
      } else {
        const mergedCourse = this.mergeCourseRecords(
          sameCourseIndices.map(idx => courses[idx])
        );
        mergedCourses.push(mergedCourse);
        sameCourseIndices.forEach(idx => usedIndices.add(idx));
      }
    }

    return mergedCourses.sort((a, b) => a.startTime.localeCompare(b.startTime));
  }

  private mergeCourseRecords(courses: CourseInfo[]): CourseInfo {
    if (courses.length === 0) {
      return courses[0];
    }

    const earliestStart = courses.reduce((min, c) => 
      c.startTime < min ? c.startTime : min, courses[0].startTime);
    const latestEnd = courses.reduce((max, c) => 
      c.endTime > max ? c.endTime : max, courses[0].endTime);

    const periods = courses.map(c => c.period).join(',');
    const mergedPeriod = this.extractMergedPeriod(periods);

    const anyCurrent = courses.some(c => c.isCurrent);
    const anyEnded = courses.every(c => c.isEnded);
    const anyUpcomingSoon = courses.some(c => c.isUpcomingSoon);
    const anyEndingSoon = courses.some(c => c.isEndingSoon);

    let status = 'upcoming';
    if (anyEnded) {
      status = 'ended';
    } else if (anyCurrent) {
      status = 'current';
    } else if (anyEndingSoon) {
      status = 'ending_soon';
    } else if (anyUpcomingSoon) {
      status = 'upcoming_soon';
    }

    return {
      name: courses[0].name,
      time: mergedPeriod,
      location: courses[0].location,
      teacher: courses[0].teacher,
      period: mergedPeriod,
      startTime: earliestStart,
      endTime: latestEnd,
      isEnded: anyEnded,
      isCurrent: anyCurrent,
      isUpcomingSoon: anyUpcomingSoon,
      isEndingSoon: anyEndingSoon,
      status: status,
      hasConflict: courses[0].hasConflict,
      conflictWith: courses[0].conflictWith
    };
  }

  private extractMergedPeriod(periods: string): string {
    const allNumbers: number[] = [];
    const periodRegex = /(\d{1,2})[\-~]?(\d{1,2})?节/g;
    let match: RegExpExecArray | null = periodRegex.exec(periods);
    
    while (match !== null) {
      const start = parseInt(match[1]);
      const end = match[2] ? parseInt(match[2]) : start;
      for (let i = start; i <= end; i++) {
        if (!allNumbers.includes(i)) {
          allNumbers.push(i);
        }
      }
      match = periodRegex.exec(periods);
    }

    if (allNumbers.length === 0) {
      return periods;
    }

    allNumbers.sort((a, b) => a - b);
    const min = allNumbers[0];
    const max = allNumbers[allNumbers.length - 1];

    return `${String(min).padStart(2, '0')}-${String(max).padStart(2, '0')}节`;
  }

  private isTimeOverlap(c1: CourseInfo, c2: CourseInfo): boolean {
    const timeToMinutes = (time: string): number => {
      const parts = time.split(':').map(Number);
      return parts[0] * 60 + parts[1];
    };

    const start1 = timeToMinutes(c1.startTime);
    const end1 = timeToMinutes(c1.endTime);
    const start2 = timeToMinutes(c2.startTime);
    const end2 = timeToMinutes(c2.endTime);

    return !(end1 <= start2 || end2 <= start1);
  }

  @Builder
  buildCourseItem(course: CourseInfo, index: number) {
    Row() {
      Column() {
        Text(`${course.startTime}-${course.endTime}`)
          .fontSize(12)
          .fontWeight(FontWeight.Bold)
          .fontColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_primary'))
          .textAlign(TextAlign.Center)
      }
      .width(80)
      .alignItems(HorizontalAlign.Center)

      Stack() {
        Column()
          .width(3)
          .height(40)
          .backgroundColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.info'))
          .borderRadius(2)

        if (course.isCurrent) {
          Column()
            .width(9)
            .height(9)
            .backgroundColor($r('app.color.success'))
            .borderRadius(5)
            .border({ width: 2, color: $r('app.color.button_text') })
            .position({ x: -3, y: 16 })
        }

        if (course.hasConflict) {
          Image($r('app.media.exclamationmark_triangle_fill'))
            .width(10)
            .height(10)
            .fillColor($r('app.color.danger'))
            .position({ x: -3, y: 16 })
        }
      }
      .width(20)
      .height(40)
      .margin({ left: 8, right: 12 })

      Column({ space: 4 }) {
        Row({ space: 8 }) {
          Text(course.name)
            .fontSize(15)
            .fontWeight(FontWeight.Medium)
            .fontColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_primary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .layoutWeight(1)

          if (course.hasConflict) {
            Row({ space: 4 }) {
              Image($r('app.media.exclamationmark_triangle_fill'))
                .width(10)
                .height(10)
                .fillColor($r('app.color.danger'))
              Text('冲突')
                .fontSize(10)
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.danger'))
            }
            .alignItems(VerticalAlign.Center)
            .backgroundColor($r('app.color.danger_light'))
            .borderRadius(4)
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
          } else if (course.isCurrent) {
            Text('进行中')
              .fontSize(10)
              .fontColor($r('app.color.success'))
              .backgroundColor($r('app.color.info_light'))
              .borderRadius(4)
              .padding({ left: 6, right: 6, top: 2, bottom: 2 })
          } else if (course.isUpcomingSoon) {
            Row({
              space: 4
            }) {
              Image($r('app.media.clock'))
                .width(12)
                .height(12)
                .fillColor($r('app.color.warning'))
              Text('即将上课')
                .fontSize(10)
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.warning'))
            }
            .alignItems(VerticalAlign.Center)
            .backgroundColor($r('app.color.info_light'))
            .borderRadius(4)
            .padding({ left: 8, right: 8, top: 3, bottom: 3 })
            .animation({
              duration: 1000,
              iterations: -1,
              curve: 'ease-in-out'
            })
            .scale({ x: 1.05, y: 1.05 })
          } else if (course.isEndingSoon) {
            Text('即将下课')
              .fontSize(10)
              .fontColor($r('app.color.primary'))
              .backgroundColor($r('app.color.info_light'))
              .borderRadius(4)
              .padding({ left: 6, right: 6, top: 2, bottom: 2 })
          }
        }
        .width('100%')

        if (course.hasConflict && course.conflictWith && course.conflictWith.length > 0) {
          Text(`与 ${course.conflictWith.join('、')} 时间冲突`)
            .fontSize(11)
            .fontColor($r('app.color.danger'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }

        Row({ space: 4 }) {
          Image($r('app.media.local_fill'))
            .width(12)
            .height(12)
            .fillColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_secondary'))
          Text(course.location || '待定')
            .fontSize(12)
            .fontColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_secondary'))
        }

        Row({ space: 4 }) {
          Image($r('app.media.person'))
            .width(12)
            .height(12)
            .fillColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_secondary'))
          Text(course.teacher || '待定')
            .fontSize(12)
            .fontColor(course.hasConflict ? $r('app.color.danger') : course.isCurrent ? $r('app.color.success') : course.isUpcomingSoon ? $r('app.color.warning') : course.isEndingSoon ? $r('app.color.primary') : $r('app.color.text_secondary'))
        }
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)
    }
    .width('100%')
    .padding({ top: 1, bottom: 6 })
  }

  @Builder
  buildQuickActionsCard() {
    Column() {
      Row() {
        Text('常用功能')
          .fontSize(17)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
      }
      .width('100%')
      .margin({ bottom: 12 })

      Flex({ wrap: FlexWrap.Wrap }) {
        this.buildActionItem('app.media.doc_text_badge_magnifyingglass', '成绩查询', () => {
          this.notificationManager.clearGradeNotification();
          router.pushUrl({ url: 'pages/GradeQueryPage' });
        }, this.notificationStatus.hasNewGrade)

        this.buildActionItem('app.media.doc_text_badge_checkmark', '培养计划', () => {
          router.pushUrl({ url: 'pages/TrainingPlanPage' });
        })

        this.buildActionItem('app.media.doc_plaintext_and_pencil', '学分进度', () => {
          this.notificationManager.clearCreditNotification();
          router.pushUrl({ url: 'pages/CreditProgressPage' });
        }, this.notificationStatus.hasNewCredit)

        this.buildActionItem('app.media.text_clipboard', '考试安排', () => {
          this.notificationManager.clearExamNotification();
          router.pushUrl({ url: 'pages/ExamPage' });
        }, this.notificationStatus.hasNewExam)

        this.buildActionItem('app.media.label', '空教室查询', () => {
          router.pushUrl({ url: 'pages/EmptyRoomPage' });
        })

        this.buildActionItem('app.media.message_on_message', 'AI学习伙伴', () => {
          router.pushUrl({ url: 'pages/AIStudyPartnerPage' });
        })
      }
      .width('100%')
    }
    .width('100%')
    .padding(20)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildActionItem(icon: string, title: string, action: () => void, showBadge: boolean = false) {
    Stack() {
      Column({ space: 8 }) {
        Image($r(icon))
          .width(32)
          .height(32)
          .fillColor(this.isDarkMode ? '#FFFFFF' : '#1C1C1E')

        Text(title)
          .fontSize(13)
          .fontColor($r('app.color.text_primary'))
          .fontWeight(FontWeight.Medium)
      }
      .width('100%')
      .padding({ top: 8, bottom: 8 })
      .onClick(action)

      if (showBadge) {
        Column()
          .width(8)
          .height(8)
          .backgroundColor('#FF3B30')
          .borderRadius(4)
          .position({ x: '72%', y: 4 })
      }
    }
    .width('25%')
    .aspectRatio(1)
  }
}

export { HomePage };
