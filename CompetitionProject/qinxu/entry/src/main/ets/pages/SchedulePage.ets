/**
 * 课表页面模块
 * 
 * 提供课程表展示和管理功能，包括：
 * - 周视图课表展示
 * - 周次切换
 * - 课程冲突检测
 * - 课程提醒设置
 * - 农历日期显示
 * 
 * @module SchedulePage
 */

import router from '@ohos.router';
import { promptAction } from '@kit.ArkUI';

import { ScheduleDataSource } from '../utils/ScheduleDataSource';
import { DataStore, Plan } from '../utils/DataStore';
import { SettingsData } from '../model/SettingsData';
import { LunarCalendar, HolidayInfo } from '../utils/LunarCalendar';
import { ThemeColors, DEFAULT_COLORS } from '../utils/ThemeColorManager';
import { LocalDataManager, ScheduleData as LocalScheduleData, CourseData } from '../utils/LocalDataManager';
import { CourseReminderService, CourseReminderInfo } from '../services/CourseReminderService';

/**
 * 培养计划课程接口
 */
interface TrainingPlanCourse {
  /** 课程名称 */
  courseName: string;
  /** 课程代码 */
  courseCode: string;
}

/**
 * 课程接口
 */
interface Course {
  /** 课程ID */
  id: string;
  /** 课程名称 */
  name: string;
  /** 授课教师 */
  teacher: string;
  /** 上课地点 */
  location: string;
  /** 星期几 */
  dayOfWeek: number;
  /** 时间段 */
  timeSlot: number;
  /** 持续节数 */
  duration: number;
  /** 周次 */
  week: number;
  /** 课程颜色 */
  color: string | Resource;
  /** 课程类型 */
  courseType?: string;
  /** 优先级 */
  priority?: number;
  /** 是否有冲突 */
  hasConflict?: boolean;
  /** 冲突课程列表 */
  conflictWith?: string[];
}

/**
 * 课程索引结构接口
 * 用于快速查询课程
 */
interface CourseIndex {
  /** 按周索引 */
  byWeek: Map<number, Course[]>;
  /** 按周+星期索引 */
  byWeekDay: Map<string, Course[]>;
  /** 按周+星期+时间段索引 */
  byWeekDaySlot: Map<string, Course>;
  /** 总课程数 */
  totalCount: number;
}

/**
 * 时间段数据接口
 */
interface TimeSlotData {
  /** 时间段ID */
  id: number;
  /** 时间范围 */
  time: string;
  /** 节次 */
  period: string;
  /** 分组索引 */
  groupIndex: number;
}

/**
 * 课表页面组件
 * 
 * 展示课程表，支持周次切换和课程详情查看
 * 
 * @example
 * ```typescript
 * // 在 MainPage 中通过 GlassMenubar 加载
 * ```
 */
@Entry
@Component
struct SchedulePage {
  @State currentWeek: number = 1;
  @State currentSemester: string = '';
  @State courses: Course[] = [];
  @State showWeekPicker: boolean = false;
  @State tempWeek: number = 1;
  @State studentId: string = '';
  @State currentDate: Date = new Date();
  @State isLoading: boolean = true;
  @State hasData: boolean = false;
  @State settings: SettingsData = new SettingsData();
  @State lunarDate: string = '';
  @State holidays: Map<string, HolidayInfo> = new Map();
  @State weekChangeAnimation: boolean = false;
  @State totalWeeksKey: number = 0;
  @State cachedDataList: LocalScheduleData[] = [];
  @StorageLink('scheduleSettingsChanged') @Watch('onSettingsChanged') scheduleSettingsChanged: boolean = false;
  @StorageLink('reminderSettingsChanged') @Watch('onReminderSettingsChanged') reminderSettingsChanged: boolean = false;
  @StorageLink('needRefreshReminders') @Watch('onNeedRefreshReminders') needRefreshReminders: boolean = false;
  @StorageLink('themeColors') themeColors: ThemeColors = DEFAULT_COLORS;
  private swiperController: SwiperController = new SwiperController();
  private weekPickerScroller: Scroller = new Scroller();
  private localDataManager: LocalDataManager = LocalDataManager.getInstance();
  private reminderService: CourseReminderService = CourseReminderService.getInstance();

  // 新增：课程索引，用于O(1)时间复杂度查询
  private courseIndex: CourseIndex = {
    byWeek: new Map(),
    byWeekDay: new Map(),
    byWeekDaySlot: new Map(),
    totalCount: 0
  };
  // 新增：当前周的课程缓存
  private currentWeekCourses: Course[] = [];
  // 新增：培养计划课程缓存
  private trainingPlanCourses: TrainingPlanCourse[] = [];

  private timeSlots: TimeSlotData[] = [
    { id: 1, time: '08:30-09:15', period: '1', groupIndex: 0 },
    { id: 2, time: '09:20-10:05', period: '2', groupIndex: 0 },
    { id: 3, time: '10:25-11:05', period: '3', groupIndex: 1 },
    { id: 4, time: '11:15-12:00', period: '4', groupIndex: 1 },
    { id: 5, time: '14:30-15:15', period: '5', groupIndex: 2 },
    { id: 6, time: '15:20-16:05', period: '6', groupIndex: 2 },
    { id: 7, time: '16:15-17:00', period: '7', groupIndex: 3 },
    { id: 8, time: '17:05-17:50', period: '8', groupIndex: 3 },
    { id: 9, time: '18:20-19:05', period: '9', groupIndex: 4 },
    { id: 10, time: '19:10-19:55', period: '10', groupIndex: 4 },
    { id: 11, time: '20:05-20:50', period: '11', groupIndex: 5 },
    { id: 12, time: '20:55-21:40', period: '12', groupIndex: 5 }
  ];

  private weekDays: string[] = ['一', '二', '三', '四', '五', '六', '日'];
  private readonly CELL_HEIGHT: number = 120;
  private readonly HEADER_HEIGHT: number = 40;
  private readonly TIME_COL_WIDTH: number = 55;
  private readonly REST_HEIGHT: number = 20;

  // 获取分组后的时间槽（每两节课一组）
  private groupedTimeSlots: TimeSlotData[][] = this.getGroupedTimeSlots();

  aboutToAppear() {
    this.studentId = AppStorage.Get('studentId') as string || '';
    this.loadSettings().then(() => {
      this.loadCachedData();
    });
    this.silentRefresh();
    this.updateLunarDate();
    this.loadHolidays();
  }

  onSettingsChanged(): void {
    console.log('SchedulePage: onSettingsChanged triggered, scheduleSettingsChanged:', this.scheduleSettingsChanged);
    if (this.scheduleSettingsChanged) {
      this.handleSettingsChange();
    }
  }

  onReminderSettingsChanged(): void {
    console.log('SchedulePage: onReminderSettingsChanged triggered, reminderSettingsChanged:', this.reminderSettingsChanged);
    if (this.reminderSettingsChanged) {
      this.handleReminderSettingsChange();
    }
  }

  onNeedRefreshReminders(): void {
    console.log('SchedulePage: onNeedRefreshReminders triggered, needRefreshReminders:', this.needRefreshReminders);
    if (this.needRefreshReminders) {
      this.handleAutoRefreshReminders();
    }
  }

  private async handleAutoRefreshReminders(): Promise<void> {
    console.log('SchedulePage: handleAutoRefreshReminders called');
    await this.setupRemindersForCurrentWeek();
    
    this.needRefreshReminders = false;
    AppStorage.SetOrCreate('needRefreshReminders', false);
    
    console.log('SchedulePage: Auto refresh reminders completed');
  }

  private async handleReminderSettingsChange(): Promise<void> {
    console.log('SchedulePage: handleReminderSettingsChange called');
    await this.setupRemindersForCurrentWeek();
    
    this.reminderSettingsChanged = false;
    AppStorage.SetOrCreate('reminderSettingsChanged', false);
    
    console.log('SchedulePage: Reminder settings updated');
  }

  private async handleSettingsChange(): Promise<void> {
    console.log('SchedulePage: handleSettingsChange called');
    await this.loadSettings();
    
    // 确保当前周数在有效范围内
    if (this.currentWeek > this.settings.totalWeeks) {
      this.currentWeek = 1;
    }
    
    // 更新总周数 key 以强制 Swiper 重新渲染
    this.totalWeeksKey = this.settings.totalWeeks;
    
    // 清除标记
    this.scheduleSettingsChanged = false;
    AppStorage.SetOrCreate('scheduleSettingsChanged', false);
    
    console.log('SchedulePage: Settings updated, totalWeeks:', this.settings.totalWeeks);
  }

  onPageShow() {
    console.log('SchedulePage: onPageShow called');

    const oldStartDate = this.settings.semesterStartDate;
    const oldTotalWeeks = this.settings.totalWeeks;
    const oldCurrentWeek = this.currentWeek;

    console.log('SchedulePage: Old settings - startDate:', oldStartDate, 'totalWeeks:', oldTotalWeeks, 'currentWeek:', oldCurrentWeek);

    this.loadSettings().then(() => {
      console.log('SchedulePage: New settings - startDate:', this.settings.semesterStartDate, 'totalWeeks:', this.settings.totalWeeks);

      const settingsChangedFlag = AppStorage.Get('scheduleSettingsChanged') as boolean;
      console.log('SchedulePage: settingsChangedFlag:', settingsChangedFlag);

      if (settingsChangedFlag) {
        AppStorage.SetOrCreate('scheduleSettingsChanged', false);
      }

      const startDateChanged = oldStartDate !== this.settings.semesterStartDate;
      const totalWeeksChanged = oldTotalWeeks !== this.settings.totalWeeks;

      console.log('SchedulePage: Changes detected - startDate:', startDateChanged, 'totalWeeks:', totalWeeksChanged);

      const newCurrentWeek = this.settings.getCurrentWeek();
      const weekChanged = newCurrentWeek !== oldCurrentWeek;

      console.log('SchedulePage: Week calculation - old:', oldCurrentWeek, 'new:', newCurrentWeek, 'changed:', weekChanged);

      const settingsChanged = settingsChangedFlag || startDateChanged || totalWeeksChanged || weekChanged;

      console.log('SchedulePage: Final settingsChanged:', settingsChanged);

      if (settingsChanged) {
        console.log('SchedulePage: Refreshing UI due to settings change');

        if (this.currentWeek > this.settings.totalWeeks) {
          this.currentWeek = 1;
        } else if (startDateChanged || weekChanged) {
          this.currentWeek = newCurrentWeek;
        }

        console.log('SchedulePage: Updated currentWeek to:', this.currentWeek);

        const newSettings = new SettingsData();
        newSettings.semesterName = this.settings.semesterName;
        newSettings.semesterStartDate = this.settings.semesterStartDate;
        newSettings.totalWeeks = this.settings.totalWeeks;
        newSettings.maxPeriods = this.settings.maxPeriods;
        newSettings.showWeekend = this.settings.showWeekend;
        newSettings.showCurrentWeek = this.settings.showCurrentWeek;
        this.settings = newSettings;

        this.currentDate = new Date();
        this.updateLunarDate();

        this.totalWeeksKey = this.settings.totalWeeks;

        if (settingsChangedFlag) {
          promptAction.showToast({ message: '课表设置已更新' });
        }
      } else {
        console.log('SchedulePage: No settings change detected, skipping refresh');
      }
    });
  }

  private async loadSettings(): Promise<void> {
    try {
      const dataStore = DataStore.getInstance();
      const loadedSettings: SettingsData | null = await dataStore.loadSettings();

      console.log('SchedulePage: loadSettings - loadedSettings:', loadedSettings);

      if (loadedSettings) {
        this.settings = loadedSettings;
        console.log('SchedulePage: loadSettings - this.settings.semesterStartDate:', this.settings.semesterStartDate);
      }
      // Update currentSemester from settings if available
      if (this.settings.semesterName) {
        this.currentSemester = this.settings.semesterName;
      }
      // 如果当前周数超过总周数，重置为第 1 周
      if (this.currentWeek > this.settings.totalWeeks) {
        this.currentWeek = 1;
      } else {
        this.currentWeek = this.settings.getCurrentWeek();
      }

      console.log('SchedulePage: loadSettings - currentWeek:', this.currentWeek);
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }

  private async loadCachedData(): Promise<void> {
    try {
      if (this.studentId) {
        const dataSource = ScheduleDataSource.getInstance();
        const scheduleData = await dataSource.loadSchedule(this.studentId);

        if (scheduleData.courses.length > 0) {
          this.courses = scheduleData.courses;

          // 保存到 LocalDataManager
          const courseDataList: CourseData[] = [];
          for (let i = 0; i < scheduleData.courses.length; i++) {
            const c = scheduleData.courses[i];
            courseDataList.push({
              id: c.id,
              name: c.name,
              teacher: c.teacher,
              location: c.location,
              dayOfWeek: c.dayOfWeek,
              period: `${c.timeSlot}-${c.timeSlot + c.duration - 1}`,
              week: c.week
            });
          }
          await this.localDataManager.saveScheduleData(courseDataList, scheduleData.semester);

          // 构建课程索引，加速后续查询
          this.buildCourseIndex();

          // 预加载培养计划课程
          await this.loadTrainingPlanCourses();

          // 设置课程类型和优先级
          this.assignCourseTypesAndPrioritiesOptimized();

          // 检测课程冲突（只标记，不阻止显示）
          this.detectCourseConflicts();

          // 预加载当前周的课程
          this.currentWeekCourses = this.getCoursesForWeekOptimized(this.currentWeek);

          if (scheduleData.semester) {
            this.currentSemester = scheduleData.semester;
            this.settings.semesterName = this.currentSemester;
            this.settings.setSemesterStartDate(this.currentSemester);
          }

          this.hasData = true;
          this.currentDate = new Date();
        }
      }
    } catch (error) {
      console.error('Failed to load cached data:', error);
    }
    this.isLoading = false;
  }

  private async silentRefresh(): Promise<void> {
    if (!this.studentId) return;

    try {
      const dataSource = ScheduleDataSource.getInstance();
      const scheduleData = await dataSource.refreshSchedule(this.studentId);

      if (scheduleData.courses.length > 0) {
        this.courses = scheduleData.courses;

        // 重建课程索引
        this.buildCourseIndex();

        // 设置课程类型和优先级（使用优化版本）
        this.assignCourseTypesAndPrioritiesOptimized();

        // 检测课程冲突（只标记，不阻止显示）
        this.detectCourseConflicts();

        // 更新当前周课程缓存
        this.currentWeekCourses = this.getCoursesForWeekOptimized(this.currentWeek);

        if (scheduleData.semester) {
          this.currentSemester = scheduleData.semester;
          const dataStore = DataStore.getInstance();
          this.settings.semesterName = this.currentSemester;
          this.settings.setSemesterStartDate(this.currentSemester);
          await dataStore.saveSettings(this.settings);
        }

        this.hasData = true;
        this.currentDate = new Date();
      }
    } catch (error) {
      console.error('SchedulePage: Silent refresh failed:', error);
    }
  }

  private async refreshData(): Promise<void> {
    if (!this.studentId) return;

    this.isLoading = true;
    try {
      const dataSource = ScheduleDataSource.getInstance();
      await dataSource.clearAllCache();

      const scheduleData = await dataSource.refreshSchedule(this.studentId);
      this.courses = scheduleData.courses;

      // 重建课程索引
      this.buildCourseIndex();

      // 设置课程类型和优先级（使用优化版本）
      this.assignCourseTypesAndPrioritiesOptimized();

      // 检测课程冲突（只标记，不阻止显示）
      this.detectCourseConflicts();

      // 更新当前周课程缓存
      this.currentWeekCourses = this.getCoursesForWeekOptimized(this.currentWeek);

      if (scheduleData.semester) {
        this.currentSemester = scheduleData.semester;
        const dataStore = DataStore.getInstance();
        this.settings.semesterName = this.currentSemester;
        this.settings.setSemesterStartDate(this.currentSemester);
        await dataStore.saveSettings(this.settings);
      }

      this.hasData = this.courses.length > 0;
      this.currentDate = new Date();
      promptAction.showToast({ message: '刷新成功' });
    } catch (error) {
      console.error('Failed to refresh data:', error);
      promptAction.showToast({ message: '刷新失败' });
    }
    this.isLoading = false;
  }

  // 新增：构建课程索引
  private buildCourseIndex(): void {
    const byWeek = new Map<number, Course[]>();
    const byWeekDay = new Map<string, Course[]>();
    const byWeekDaySlot = new Map<string, Course>();

    for (let i = 0; i < this.courses.length; i++) {
      const course = this.courses[i];

      // 按周索引
      if (!byWeek.has(course.week)) {
        byWeek.set(course.week, []);
      }
      byWeek.get(course.week)!.push(course);

      // 按周+星期索引
      const weekDayKey = `${course.week}-${course.dayOfWeek}`;
      if (!byWeekDay.has(weekDayKey)) {
        byWeekDay.set(weekDayKey, []);
      }
      byWeekDay.get(weekDayKey)!.push(course);

      // 按周+星期+时间段索引（只记录起始时间段）
      const weekDaySlotKey = `${course.week}-${course.dayOfWeek}-${course.timeSlot}`;
      byWeekDaySlot.set(weekDaySlotKey, course);
    }

    this.courseIndex = {
      byWeek,
      byWeekDay,
      byWeekDaySlot,
      totalCount: this.courses.length
    };

    console.log(`SchedulePage: Built course index with ${this.courses.length} courses`);
  }

  // 优化版本：使用索引获取某周的课程（O(1)时间复杂度）
  private getCoursesForWeekOptimized(week: number): Course[] {
    if (this.courseIndex.byWeek.has(week)) {
      return this.courseIndex.byWeek.get(week) || [];
    }
    return [];
  }

  // 兼容旧版本：线性遍历获取某周的课程
  private getCoursesForWeek(week: number): Course[] {
    // 如果索引存在且有效，使用索引查询
    if (this.courseIndex.totalCount > 0) {
      return this.getCoursesForWeekOptimized(week);
    }
    // 否则使用线性遍历（兼容旧代码）
    const result: Course[] = [];
    for (let i = 0; i < this.courses.length; i++) {
      if (this.courses[i].week === week) {
        result.push(this.courses[i]);
      }
    }
    return result;
  }

  // 获取指定周、指定星期的日期
  private getDateForDay(dayOfWeek: number, week: number): string {
    if (!this.settings.semesterStartDate) {
      const today = new Date();
      const currentDayOfWeek = today.getDay() || 7;
      const dayDiff = dayOfWeek - currentDayOfWeek;
      const targetDate = new Date(today);
      targetDate.setDate(today.getDate() + dayDiff);
      console.log('SchedulePage: getDateForDay - no semesterStartDate, using calculated date for day:', dayOfWeek);
      return `${targetDate.getDate()}日`;
    }

    const startDate = new Date(this.settings.semesterStartDate);
    const targetDate = new Date(startDate);
    targetDate.setDate(startDate.getDate() + ((week - 1) * 7) + (dayOfWeek - 1));

    console.log('SchedulePage: getDateForDay - dayOfWeek:', dayOfWeek, 'week:', week, 'startDate:', this.settings.semesterStartDate, 'targetDate:', targetDate.toDateString());

    return `${targetDate.getDate()}日`;
  }

  private getFullDateForDay(dayOfWeek: number, week: number): string {
    if (!this.settings.semesterStartDate) {
      const today = new Date();
      return `${today.getMonth() + 1}月${today.getDate()}日`;
    }

    const startDate = new Date(this.settings.semesterStartDate);

    // 开学日期就是第一周的第一天
    // 第N周第D天 = 开学日期 + (N-1)*7 + (D-1)天
    const targetDate = new Date(startDate);
    targetDate.setDate(startDate.getDate() + ((week - 1) * 7) + (dayOfWeek - 1));

    return `${targetDate.getMonth() + 1}月${targetDate.getDate()}日`;
  }

  private getCurrentWeekDay(): number {
    return new Date().getDay() || 7;
  }

  private isToday(week: number, dayOfWeek: number): boolean {
    if (!this.settings.semesterStartDate) {
      return false;
    }

    const startDate = new Date(this.settings.semesterStartDate);
    const targetDate = new Date(startDate);
    targetDate.setDate(startDate.getDate() + ((week - 1) * 7) + (dayOfWeek - 1));

    const today = new Date();
    return targetDate.getDate() === today.getDate() &&
           targetDate.getMonth() === today.getMonth() &&
           targetDate.getFullYear() === today.getFullYear();
  }

  private getMonthForWeek(week: number): string {
    if (!this.settings.semesterStartDate) {
      const targetDate = new Date();
      const year = targetDate.getFullYear();
      const month = targetDate.getMonth() + 1;
      return `${year}年${month}月`;
    }

    const startDate = new Date(this.settings.semesterStartDate);

    // 开学日期就是第一周的第一天
    // 第N周的第一天 = 开学日期 + (N-1)*7天
    const targetDate = new Date(startDate);
    targetDate.setDate(startDate.getDate() + ((week - 1) * 7));

    const year = targetDate.getFullYear();
    const month = targetDate.getMonth() + 1;

    return `${year}年${month}月`;
  }

  private formatSemesterDisplay(semester: string): string {
    if (!semester) return '加载中...';
    if (semester.includes('-')) {
      const parts = semester.split('-');
      if (parts.length === 3) {
        const term = parts[2] === '1' ? '第一学期' : '第二学期';
        return `${parts[0]}-${parts[1]}学年 ${term}`;
      }
    }
    return semester;
  }

  private updateLunarDate(): void {
    this.lunarDate = LunarCalendar.getLunarDateString(this.currentDate);
  }

  private async loadHolidays(): Promise<void> {
    const year = this.currentDate.getFullYear();
    this.holidays = await LunarCalendar.getHolidays(year);
  }

  private isTodayHoliday(): boolean {
    return LunarCalendar.isHoliday(this.currentDate, this.holidays);
  }

  private getHolidayName(): string {
    return LunarCalendar.getHolidayName(this.currentDate, this.holidays);
  }

  // 获取分组后的时间槽（每两节课一组）
  private getGroupedTimeSlots(): TimeSlotData[][] {
    const groups: TimeSlotData[][] = [];
    const groupMap = new Map<number, TimeSlotData[]>();

    // 按groupIndex分组
    this.timeSlots.forEach(slot => {
      if (!groupMap.has(slot.groupIndex)) {
        groupMap.set(slot.groupIndex, []);
      }
      groupMap.get(slot.groupIndex)!.push(slot);
    });

    // 按groupIndex排序并返回
    const sortedGroups = Array.from(groupMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(entry => entry[1]);

    return sortedGroups;
  }

  private async loadTrainingPlanCourses(): Promise<void> {
    try {
      const dataStore = DataStore.getInstance();
      const plans: Plan[] = await dataStore.loadPlans();

      this.trainingPlanCourses = [];
      for (let i = 0; i < plans.length; i++) {
        this.trainingPlanCourses.push({
          courseName: plans[i].courseName,
          courseCode: plans[i].courseCode
        });
      }
      console.log(`SchedulePage: Loaded ${this.trainingPlanCourses.length} training plan courses`);
    } catch (error) {
      console.error('Failed to load training plan courses:', error);
      this.trainingPlanCourses = [];
    }
  }

  // 优化版本：使用内存中的培养计划课程数据（无需重复加载）
  private assignCourseTypesAndPrioritiesOptimized(): void {
    // 创建课程名称到培养计划的快速查找Set
    const trainingPlanNames = new Set<string>();
    const trainingPlanCodes = new Set<string>();

    for (let i = 0; i < this.trainingPlanCourses.length; i++) {
      const tc = this.trainingPlanCourses[i];
      trainingPlanNames.add(tc.courseName);
      trainingPlanCodes.add(tc.courseCode);
    }

    for (let i = 0; i < this.courses.length; i++) {
      const course = this.courses[i];
      const courseName = course.name.toLowerCase();

      // 使用Set进行O(1)查找
      const isInTrainingPlan = trainingPlanNames.has(course.name) || trainingPlanCodes.has(course.id);

      if (isInTrainingPlan) {
        course.courseType = '培养计划';
        course.priority = 1;
      } else if (courseName.includes('重修')) {
        course.courseType = '重修';
        course.priority = 2;
      } else if (courseName.includes('辅修')) {
        course.courseType = '辅修';
        course.priority = 3;
      } else {
        course.courseType = '其他';
        course.priority = 4;
      }
    }
  }

  // 兼容旧版本：异步获取培养计划课程
  private async assignCourseTypesAndPriorities(): Promise<void> {
    await this.loadTrainingPlanCourses();
    this.assignCourseTypesAndPrioritiesOptimized();
  }

  // 兼容旧版本：获取培养计划课程
  private async getTrainingPlanCourses(): Promise<TrainingPlanCourse[]> {
    if (this.trainingPlanCourses.length === 0) {
      await this.loadTrainingPlanCourses();
    }
    return this.trainingPlanCourses;
  }

  // 优化版本：使用已有索引检测课程冲突
  private detectCourseConflicts(): void {
    // 重置冲突状态
    for (let i = 0; i < this.courses.length; i++) {
      const course = this.courses[i];
      course.hasConflict = false;
      course.conflictWith = [];
    }

    // 使用已有的courseIndex.byWeekDay进行冲突检测
    this.courseIndex.byWeekDay.forEach((courses) => {
      // 如果当天课程少于2个，不可能有冲突
      if (courses.length < 2) return;

      for (let i = 0; i < courses.length; i++) {
        for (let j = i + 1; j < courses.length; j++) {
          const c1 = courses[i];
          const c2 = courses[j];

          if (this.isTimeOverlap(c1, c2)) {
            c1.hasConflict = true;
            c2.hasConflict = true;

            if (!c1.conflictWith!.includes(c2.id)) {
              c1.conflictWith!.push(c2.id);
            }
            if (!c2.conflictWith!.includes(c1.id)) {
              c2.conflictWith!.push(c1.id);
            }
          }
        }
      }
    });
  }

  // 检查两个课程的时间是否重叠
  private isTimeOverlap(c1: Course, c2: Course): boolean {
    if (c1.week !== c2.week || c1.dayOfWeek !== c2.dayOfWeek) {
      return false;
    }

    if (c1.id === c2.id || c1.name === c2.name) {
      return false;
    }

    const c1Start = c1.timeSlot;
    const c1End = c1.timeSlot + c1.duration;
    const c2Start = c2.timeSlot;
    const c2End = c2.timeSlot + c2.duration;

    return !(c1End <= c2Start || c2End <= c1Start);
  }

  // 根据文本长度获取自适应字体大小
  private getAdaptiveFontSize(text: string, baseSize: number, minSize: number): number {
    const maxLength = 20; // 最大字符长度
    if (text.length <= maxLength) {
      return baseSize;
    }
    const excessLength = text.length - maxLength;
    const sizeReduction = Math.min(excessLength * 0.3, baseSize - minSize);
    return baseSize - sizeReduction;
  }

  // 根据文本长度获取自适应字体粗细
  private getAdaptiveFontWeight(text: string): number {
    if (text.length > 15) {
      return FontWeight.Normal;
    }
    return FontWeight.Medium;
  }

  // 获取优先级课程
  private getPriorityCourses(week: number): Course[] {
    const weekCourses = this.getCoursesForWeek(week);
    return weekCourses
      .filter((c: Course) => c.priority === 1)
      .sort((a: Course, b: Course) => {
        if (a.dayOfWeek !== b.dayOfWeek) {
          return a.dayOfWeek - b.dayOfWeek;
        }
        return a.timeSlot - b.timeSlot;
      });
  }

  private convertToReminderInfo(course: Course): CourseReminderInfo {
    const startSlot = this.timeSlots.find(slot => slot.id === course.timeSlot);
    const endSlot = this.timeSlots.find(slot => slot.id === course.timeSlot + course.duration - 1);
    const startTime = startSlot ? startSlot.time.split('-')[0] : '08:30';
    const endTime = endSlot ? endSlot.time.split('-')[1] : '09:15';

    return {
      courseName: course.name,
      location: course.location,
      teacher: course.teacher,
      dayOfWeek: course.dayOfWeek,
      startTime: startTime,
      endTime: endTime,
      week: course.week
    };
  }

  private async setupRemindersForCurrentWeek(): Promise<void> {
    const settings = this.reminderService.getSettings();
    if (!settings.enabled) {
      return;
    }

    const weekCourses = this.getCoursesForWeek(this.currentWeek);
    const reminderInfos: CourseReminderInfo[] = [];

    for (const course of weekCourses) {
      reminderInfos.push(this.convertToReminderInfo(course));
    }

    // 获取下一周课程数据（用于跨周明日提醒）
    const nextWeekCourses = this.getCoursesForWeek(this.currentWeek + 1);
    const nextWeekReminderInfos: CourseReminderInfo[] = [];

    for (const course of nextWeekCourses) {
      nextWeekReminderInfos.push(this.convertToReminderInfo(course));
    }

    await this.reminderService.publishWeeklyReminders(reminderInfos, nextWeekReminderInfos);
    console.info(`SchedulePage: Set up ${reminderInfos.length} reminders for week ${this.currentWeek}`);
  }

  build() {
    Stack() {
      Column() {
        Column()
          .width('100%')
          .height(44)

        this.buildHeader()
        this.buildScheduleContent()
      }
      .width('100%')
      .height('100%')
      .backgroundColor($r('app.color.background_primary'))

      if (this.showWeekPicker) {
        this.buildWeekPickerDialog()
      }
    }
    .width('100%')
    .height('100%')
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildHeader() {
    Column() {
      Row() {
        Column() {
          Row({ space: 8 }) {
            Text(`${this.currentDate.getMonth() + 1}月${this.currentDate.getDate()}日`)
              .fontSize($r('sys.float.ohos_id_text_size_headline8'))
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))

            Row({ space: 4 }) {
              Text(`第`)
                .fontSize(14)
                .fontColor($r('app.color.text_primary'))

              Text(`${this.currentWeek}`)
                .fontSize(14)
                .fontColor(this.currentWeek === this.settings.getCurrentWeek() ? this.themeColors.scheduleWeek : $r('app.color.text_primary'))
                .fontWeight(FontWeight.Medium)

              Text(`周`)
                .fontSize(14)
                .fontColor($r('app.color.text_primary'))
            }
            .margin({ left: 8 })
            .padding({ left: 8, right: 8, top: 4, bottom: 4 })
            .onClick(() => {
              this.tempWeek = this.currentWeek;
              this.showWeekPicker = true;
            })
          }
          .margin({ top: 4 })

          Row({ space: 6 }) {
            Text(this.lunarDate)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))

            if (this.isTodayHoliday()) {
              Text(`· ${this.getHolidayName()}`)
                .fontSize(12)
                .fontColor('#E84026')
                .fontWeight(FontWeight.Medium)
            }
          }
          .margin({ top: 2 })

          Text(this.getMonthForWeek(this.currentWeek))
            .fontSize(12)
            .fontColor(this.themeColors.scheduleMonth)
            .fontWeight(FontWeight.Medium)
            .margin({ top: 4 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        Row({ space: 16 }) {
          Image($r('app.media.arrow_clockwise'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_primary'))
            .onClick(() => {
              this.refreshData();
            })

          Image($r('app.media.arrow_counterclockwise_clock'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_primary'))
            .onClick(() => {
              this.showCacheHistoryDialog();
            })

          Image($r('app.media.gearshape'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_primary'))
            .onClick(async () => {
              await router.pushUrl({ url: 'pages/ScheduleSettingsPage' });
              // 从设置页面返回后立即刷新
              this.loadSettings();
              if (this.currentWeek > this.settings.totalWeeks) {
                this.currentWeek = 1;
              }
              this.silentRefresh();
              this.updateLunarDate();
            })
        }
      }
      .width('100%')
      
      // 学期信息移到设置和刷新按钮下方
      Text(this.formatSemesterDisplay(this.currentSemester))
        .fontSize($r('sys.float.ohos_id_text_size_body2'))
        .fontColor(this.themeColors.scheduleSemester)
        .fontWeight(FontWeight.Medium)
        .margin({ top: 8 })
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 8, bottom: 8 })
    .backgroundColor($r('app.color.background_primary'))
  }

  @Builder
  buildScheduleContent() {
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
    } else if (!this.hasData) {
      Column() {
        Image($r('app.media.info_circle'))
          .width(80)
          .height(80)
          .fillColor($r('app.color.text_tertiary'))
        Text('暂无课表数据')
          .fontSize(16)
          .fontColor($r('app.color.text_secondary'))
          .margin({ top: 16 })
        Text('请先登录并同步数据')
          .fontSize(14)
          .fontColor($r('app.color.text_tertiary'))
          .margin({ top: 8 })
      }
      .width('100%')
      .layoutWeight(1)
      .justifyContent(FlexAlign.Center)
    } else {
      Column() {
        // 固定的星期和日期头部
        Row() {
          Column() {
            Text('节')
              .fontSize(11)
              .fontColor($r('app.color.text_secondary'))
          }
          .width(this.TIME_COL_WIDTH)
          .height(this.HEADER_HEIGHT)
          .justifyContent(FlexAlign.Center)
          .borderRadius(8)
          .backgroundColor($r('app.color.background_tertiary'))

          ForEach(this.settings.showWeekend ? [1, 2, 3, 4, 5, 6, 7] : [1, 2, 3, 4, 5], (dayIndex: number) => {
            Column() {
              Text(`周${this.weekDays[dayIndex - 1]}`)
                .fontSize(11)
                .fontColor(this.settings.showCurrentWeek && dayIndex === this.getCurrentWeekDay() && this.currentWeek === this.settings.getCurrentWeek() ? this.themeColors.highlightColor : $r('app.color.text_secondary'))
              Text(`${this.getDateForDay(dayIndex, this.currentWeek)}`)
                .fontSize(10)
                .fontWeight(FontWeight.Medium)
                .fontColor(this.settings.showCurrentWeek && dayIndex === this.getCurrentWeekDay() && this.currentWeek === this.settings.getCurrentWeek() ? this.themeColors.highlightColor : $r('app.color.text_primary'))
                .margin({ top: 2 })
                .animation({
                  duration: 300,
                  curve: Curve.FastOutSlowIn,
                  delay: 0,
                  iterations: 1
                })
            }
            .layoutWeight(1)
            .height(this.HEADER_HEIGHT)
            .justifyContent(FlexAlign.Center)
            .borderRadius(8)
            .margin({ left: 3, right: 3 })
            .backgroundColor(this.settings.showCurrentWeek && dayIndex === this.getCurrentWeekDay() && this.currentWeek === this.settings.getCurrentWeek() ? this.themeColors.highlightColor + '20' : $r('app.color.background_primary'))
          })
        }
        .width('100%')

        Scroll() {
          Row() {
            Column() {
              ForEach(this.groupedTimeSlots, (group: TimeSlotData[], index?: number) => {
                Column() {
                  Text(`${group[0].period}-${group[1].period}`)
                    .fontSize(11)
                    .fontWeight(FontWeight.Bold)
                    .fontColor($r('app.color.text_primary'))
                  Text(`${group[0].time.split('-')[0]}`)
                    .fontSize(9)
                    .fontColor($r('app.color.text_secondary'))
                    .margin({ top: 4 })
                  Text('-')
                    .fontSize(9)
                    .fontColor($r('app.color.text_secondary'))
                  Text(`${group[1].time.split('-')[1]}`)
                    .fontSize(9)
                    .fontColor($r('app.color.text_secondary'))
                }
                .width('100%')
                .height(this.CELL_HEIGHT)
                .justifyContent(FlexAlign.Center)
                .alignItems(HorizontalAlign.Center)
                .backgroundColor($r('app.color.background_primary'))
                .borderRadius(8)

                if (index === 1) {
                  Row()
                    .width('100%')
                    .height(this.REST_HEIGHT)
                    .backgroundColor($r('app.color.background_tertiary'))
                }

                if (index === 3) {
                  Row()
                    .width('100%')
                    .height(this.REST_HEIGHT)
                    .backgroundColor($r('app.color.background_tertiary'))
                }
              })
            }
            .width(this.TIME_COL_WIDTH)
            .alignItems(HorizontalAlign.Center)
            .padding({ bottom: 60 })

            Swiper(this.swiperController) {
              ForEach(this.getWeekArray(), (week: number) => {
                Stack({ alignContent: Alignment.TopStart }) {
                  this.buildBackgroundGrid(week)
                  
                  Row() {
                    ForEach(this.settings.showWeekend ? [1, 2, 3, 4, 5, 6, 7] : [1, 2, 3, 4, 5], (dayIndex: number) => {
                      Stack({ alignContent: Alignment.TopStart }) {
                        ForEach(this.getCoursesForDayAndWeek(dayIndex, week), (course: Course) => {
                          this.buildCourseCardInColumn(course)
                        }, (course: Course) => `${course.id}-${course.week}-${course.dayOfWeek}-${course.timeSlot}`)
                      }
                      .layoutWeight(1)
                      .height('100%')
                      .margin({ left: 3, right: 3 })
                    })
                  }
                  .width('100%')
                  .height('100%')
                  .padding({ bottom: 60 })
                }
                .width('100%')
                .layoutWeight(1)
              }, (week: number) => `${this.totalWeeksKey}-${week}`)
            }
            .index(this.currentWeek - 1)
            .autoPlay(false)
            .indicator(false)
            .loop(false)
            .duration(400)
            .curve(Curve.FastOutSlowIn)
            .onChange((index: number) => {
              this.currentWeek = index + 1;
              this.currentWeekCourses = this.getCoursesForWeekOptimized(this.currentWeek);
            })
            .layoutWeight(1)
          }
        }
        .scrollBar(BarState.Auto)
        .edgeEffect(EdgeEffect.Spring)
        .width('100%')
        .layoutWeight(1)
      }
      .width('100%')
      .layoutWeight(1)
      .margin({ top: 4 })
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius({ topLeft: 20, topRight: 20 })
    }
  }

  @Builder
  buildBackgroundGrid(week: number) {
    Column() {
      ForEach(this.groupedTimeSlots, (group: TimeSlotData[], index?: number) => {
        Column() {
          Row() {
            ForEach(this.settings.showWeekend ? [1, 2, 3, 4, 5, 6, 7] : [1, 2, 3, 4, 5], (dayIndex: number) => {
              Column()
                .layoutWeight(1)
                .height(this.CELL_HEIGHT)
                .backgroundColor(this.isToday(week, dayIndex) ? this.themeColors.highlightColor + '20' : $r('app.color.background_primary'))
                .border({ width: 0.5, color: $r('app.color.divider') })
                .borderRadius(8)
                .margin({ left: 3, right: 3 })
            })
          }
          .width('100%')
          .height(this.CELL_HEIGHT)
          
          if (index === 1) {
            this.buildRestDivider('午休', '12:00-14:30')
          }
          
          if (index === 3) {
            this.buildRestDivider('晚休', '17:50-18:20')
          }
        }
      })
    }
    .width('100%')
    .padding({ bottom: 60 })
  }

  @Builder
  buildRestDivider(title: string, time: string) {
    Row() {
      Text(title)
        .fontSize(10)
        .fontColor($r('app.color.text_tertiary'))
      
      Text(time)
        .fontSize(9)
        .fontColor($r('app.color.text_tertiary'))
        .margin({ left: 8 })
    }
    .width('100%')
    .height(this.REST_HEIGHT)
    .justifyContent(FlexAlign.Center)
    .backgroundColor($r('app.color.background_tertiary'))
  }

  private getCoursesForDayAndWeek(day: number, week: number): Course[] {
    const weekDayKey = `${week}-${day}`;
    const courses = this.courseIndex.byWeekDay.get(weekDayKey) || [];
    const uniqueCourses: Course[] = [];
    const seenSlots = new Set<number>();
    
    for (const course of courses) {
      if (!seenSlots.has(course.timeSlot)) {
        uniqueCourses.push(course);
        seenSlots.add(course.timeSlot);
      }
    }
    
    return uniqueCourses.sort((a, b) => a.timeSlot - b.timeSlot);
  }

  @Builder
  buildCourseCardInColumn(course: Course) {
    Stack() {
      Column({ space: 2 }) {
        Column({ space: 1 }) {
          Text(course.name)
            .fontSize(this.getAdaptiveNameFontSize(course.name, course.duration))
            .fontWeight(FontWeight.Bold)
            .fontColor('#FFFFFF')
            .maxLines(3)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')
          
          Text(this.formatLocation(course.location))
            .fontSize(this.getAdaptiveLocationFontSize(course.location, course.duration))
            .fontWeight(FontWeight.Normal)
            .fontColor('rgba(255,255,255,0.95)')
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)
        
        if (course.duration > 2) {
          Row({ space: 4 }) {
            Text(`${course.duration}节`)
              .fontSize(this.getInfoFontSize(course.duration))
              .fontColor('rgba(255,255,255,0.8)')
              .padding({ left: 4, right: 4, top: 1, bottom: 1 })
              .backgroundColor('rgba(255,255,255,0.2)')
              .borderRadius(3)
            
            if (course.teacher && course.teacher.length > 0) {
              Text(course.teacher)
                .fontSize(this.getInfoFontSize(course.duration))
                .fontColor('rgba(255,255,255,0.8)')
                .layoutWeight(1)
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
            }
          }
          .width('100%')
          .alignItems(VerticalAlign.Center)
          .margin({ top: 2 })
        }
        
        if (course.hasConflict && course.conflictWith && course.conflictWith.length > 0) {
          Divider()
            .strokeWidth(1)
            .color('rgba(255,255,255,0.5)')
            .margin({ top: 2, bottom: 2 })
          
          if (course.conflictWith.length === 1) {
            Column({ space: 1 }) {
              Text(this.getConflictCourseName(course.conflictWith[0]))
                .fontSize(this.getInfoFontSize(course.duration) + 0.5)
                .fontWeight(FontWeight.Bold)
                .fontColor('rgba(255,255,255,0.9)')
                .maxLines(2)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
                .width('100%')
              
              Text(this.getConflictCourseLocation(course.conflictWith[0]))
                .fontSize(this.getInfoFontSize(course.duration))
                .fontColor('rgba(255,255,255,0.8)')
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
                .width('100%')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Start)
          } else {
            Column({ space: 1 }) {
              ForEach(this.getSortedConflictCourseIds(course.conflictWith).slice(0, 2), (conflictId: string) => {
                Column({ space: 1 }) {
                  Text(this.getConflictCourseName(conflictId))
                    .fontSize(this.getInfoFontSize(course.duration))
                    .fontWeight(FontWeight.Bold)
                    .fontColor('rgba(255,255,255,0.9)')
                    .maxLines(2)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                    .width('100%')
                  
                  Text(this.getConflictCourseLocation(conflictId))
                    .fontSize(this.getInfoFontSize(course.duration) - 0.5)
                    .fontColor('rgba(255,255,255,0.8)')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                    .width('100%')
                }
                .width('100%')
                .alignItems(HorizontalAlign.Start)
                .margin({ top: 2 })
              })
              
              if (course.conflictWith.length > 2) {
                Text(`还有${course.conflictWith.length - 2}门课程冲突`)
                  .fontSize(this.getInfoFontSize(course.duration) - 0.5)
                  .fontColor('rgba(255,255,255,0.7)')
                  .margin({ top: 2 })
              }
            }
            .width('100%')
            .alignItems(HorizontalAlign.Start)
          }
        }
      }
      .width('100%')
      .height('100%')
      .padding(6)
      .justifyContent(FlexAlign.Start)
      
      if (course.hasConflict) {
        Image($r('app.media.exclamationmark_triangle_fill'))
          .width(10)
          .height(10)
          .position({ x: 'calc(100% - 12px)', y: 'calc(100% - 12px)' })
      }
    }
    .width('100%')
    .height(this.calculateCourseCardHeight(course))
    .backgroundColor(course.color)
    .borderRadius(8)
    .margin({ top: this.calculateCourseTopOffset(course) })
    .shadow({
      radius: 4,
      color: $r('app.color.overlay'),
      offsetX: 0,
      offsetY: 2
    })
    .onClick(() => {
      this.showCourseDetailDialog(course, course.dayOfWeek);
    })
    .animation({
      duration: 300,
      curve: Curve.FastOutSlowIn,
      delay: 0,
      iterations: 1
    })
  }

  @Builder
  buildCourseCardAbsolute(course: Course, week: number) {
    Stack() {
      Column({ space: 3 }) {
        Column({ space: 1 }) {
          Text(course.name)
            .fontSize(8.5)
            .fontWeight(FontWeight.Bold)
            .fontColor('#FFFFFF')
          
          Text(course.location)
            .fontSize(8)
            .fontWeight(FontWeight.Normal)
            .fontColor('rgba(255,255,255,0.95)')
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)
        
        if (course.duration > 2) {
          Row({ space: 4 }) {
            Text(`${course.duration}节`)
              .fontSize(7)
              .fontColor('rgba(255,255,255,0.8)')
              .padding({ left: 4, right: 4, top: 1, bottom: 1 })
              .backgroundColor('rgba(255,255,255,0.2)')
              .borderRadius(3)
            
            if (course.teacher && course.teacher.length > 0) {
              Text(course.teacher)
                .fontSize(7)
                .fontColor('rgba(255,255,255,0.8)')
                .layoutWeight(1)
            }
          }
          .width('100%')
          .alignItems(VerticalAlign.Center)
          .margin({ top: 2 })
        }
        
        if (course.hasConflict && course.conflictWith && course.conflictWith.length > 0) {
          Divider()
            .strokeWidth(1)
            .color('rgba(255,255,255,0.5)')
            .margin({ top: 2, bottom: 2 })
          
          if (course.conflictWith.length === 1) {
            Column({ space: 1 }) {
              Text(this.getConflictCourseName(course.conflictWith[0]))
                .fontSize(7.5)
                .fontWeight(FontWeight.Bold)
                .fontColor('rgba(255,255,255,0.9)')
              
              Text(this.getConflictCourseLocation(course.conflictWith[0]))
                .fontSize(6.5)
                .fontColor('rgba(255,255,255,0.8)')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Start)
          } else {
            Column({ space: 1 }) {
              Text(`与${course.conflictWith.length}门课程冲突`)
                .fontSize(7)
                .fontColor('rgba(255,255,255,0.9)')
                .fontWeight(FontWeight.Bold)
              
              Text('点击查看详情')
                .fontSize(6.5)
                .fontColor('rgba(255,255,255,0.8)')
            }
            .width('100%')
            .alignItems(HorizontalAlign.Start)
          }
        }
      }
      .width('100%')
      .height('100%')
      .padding(8)
      .justifyContent(FlexAlign.Start)
      
      if (course.hasConflict) {
        Image($r('app.media.exclamationmark_triangle_fill'))
          .width(10)
          .height(10)
          .position({ x: 'calc(100% - 12px)', y: 'calc(100% - 12px)' })
      }
    }
    .width(this.calculateCourseWidth())
    .height(this.calculateCourseCardHeight(course))
    .backgroundColor(course.color)
    .borderRadius(8)
    .margin({ left: this.calculateCourseLeftMargin(course.dayOfWeek), top: this.calculateCourseTopOffset(course) })
    .shadow({
      radius: 4,
      color: $r('app.color.overlay'),
      offsetX: 0,
      offsetY: 2
    })
    .onClick(() => {
      this.showCourseDetailDialog(course, course.dayOfWeek);
    })
    .animation({
      duration: 300,
      curve: Curve.FastOutSlowIn,
      delay: 0,
      iterations: 1
    })
  }

  private calculateCourseWidth(): string {
    const dayCount = this.settings.showWeekend ? 7 : 5;
    return `calc((100% - ${dayCount * 6}px) / ${dayCount})`;
  }

  private calculateCourseLeftMargin(dayOfWeek: number): number {
    const dayCount = this.settings.showWeekend ? 7 : 5;
    const dayIndex = this.settings.showWeekend ? dayOfWeek - 1 : Math.min(dayOfWeek - 1, 4);
    return dayIndex * 0;
  }

  private calculateCourseCardHeight(course: Course): number {
    const singlePeriodHeight = this.CELL_HEIGHT / 2;
    return singlePeriodHeight * course.duration;
  }

  private calculateCourseTopOffset(course: Course): number {
    const groupIndex = Math.floor((course.timeSlot - 1) / 2);
    let offset = groupIndex * this.CELL_HEIGHT;
    
    if (groupIndex >= 2) {
      offset += this.REST_HEIGHT;
    }
    if (groupIndex >= 4) {
      offset += this.REST_HEIGHT;
    }
    
    return offset;
  }

  private getConflictCourses(course: Course): Course[] {
    if (!course.conflictWith || course.conflictWith.length === 0) {
      return [];
    }
    const conflictCourses: Course[] = [];
    for (const conflictId of course.conflictWith) {
      const conflictCourse = this.courses.find((c: Course) => c.id === conflictId);
      if (conflictCourse) {
        conflictCourses.push(conflictCourse);
      }
    }
    return conflictCourses.sort((a, b) => (a.priority || 4) - (b.priority || 4));
  }

  private getConflictCourseName(conflictId: string): string {
    const conflictCourse = this.courses.find((c: Course) => c.id === conflictId);
    return conflictCourse ? conflictCourse.name : '未知课程';
  }

  private getConflictCourseLocation(conflictId: string): string {
    const conflictCourse = this.courses.find((c: Course) => c.id === conflictId);
    return conflictCourse ? this.formatLocation(conflictCourse.location) : '未知地点';
  }

  private getSortedConflictCourseIds(conflictIds: string[]): string[] {
    return conflictIds.sort((a, b) => {
      const courseA = this.courses.find((c: Course) => c.id === a);
      const courseB = this.courses.find((c: Course) => c.id === b);
      const priorityA = courseA?.priority || 4;
      const priorityB = courseB?.priority || 4;
      return priorityA - priorityB;
    });
  }

  private formatLocation(location: string): string {
    const match = location.match(/[A-Za-z]+\d+[A-Za-z\d]*/);
    return match ? match[0] : location.replace(/[^\w\d]/g, '').trim();
  }

  private getNameFontSize(duration: number): number {
    if (duration >= 4) return 11;
    if (duration === 3) return 10;
    return 8.5;
  }

  private getAdaptiveNameFontSize(name: string, duration: number): number {
    const baseSize = this.getNameFontSize(duration);
    const minLength = 7;
    if (name.length <= 4) {
      return baseSize;
    } else if (name.length <= 6) {
      return Math.max(minLength, baseSize - 0.5);
    } else if (name.length <= 8) {
      return Math.max(minLength, baseSize - 1);
    } else if (name.length <= 10) {
      return Math.max(minLength, baseSize - 1.5);
    } else {
      return Math.max(minLength, baseSize - 2);
    }
  }

  private getLocationFontSize(duration: number): number {
    if (duration >= 4) return 10;
    if (duration === 3) return 9;
    return 8;
  }

  private getAdaptiveLocationFontSize(location: string, duration: number): number {
    const baseSize = this.getLocationFontSize(duration);
    const minLength = 6;
    if (location.length <= 6) {
      return baseSize;
    } else if (location.length <= 10) {
      return Math.max(minLength, baseSize - 0.5);
    } else {
      return Math.max(minLength, baseSize - 1);
    }
  }

  private getInfoFontSize(duration: number): number {
    if (duration >= 4) return 8;
    if (duration === 3) return 7.5;
    return 7;
  }

  private showCourseDetailDialog(course: Course, dayIndex: number): void {
    const timeSlotGroup = this.timeSlots.filter(slot => slot.id === course.timeSlot);
    const group: TimeSlotData[] = timeSlotGroup.length > 0 ? 
      this.groupedTimeSlots[Math.floor((course.timeSlot - 1) / 2)] : 
      this.groupedTimeSlots[0];
    
    this.showConflictDialog(course, dayIndex, group);
  }

  @Builder
  buildWeekPickerDialog() {
    Column() {
      Column({ space: 6 }) {
        Text('选择周数')
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .margin({ bottom: 2 })

        Scroll(this.weekPickerScroller) {
          Grid() {
            ForEach(this.getWeekArray(), (week: number) => {
              GridItem() {
                Text(this.getWeekDisplayText(week))
                  .width('100%')
                  .height(32)
                  .fontSize(11)
                  .fontColor(this.tempWeek === week ? $r('app.color.background_primary') : $r('app.color.text_primary'))
                  .fontWeight(this.tempWeek === week ? FontWeight.Bold : FontWeight.Normal)
                  .backgroundColor(this.tempWeek === week ? $r('app.color.primary') : $r('app.color.background_tertiary'))
                  .borderRadius(5)
                  .textAlign(TextAlign.Center)
                  .onClick(() => {
                    this.tempWeek = week;
                    this.confirmWeekSelection();
                  })
              }
            }, (week: number) => `${this.totalWeeksKey}-${week}`)
          }
          .columnsTemplate('1fr 1fr 1fr 1fr')
          .rowsGap(5)
          .columnsGap(5)
          .width('100%')
          .scrollBar(BarState.Off)
          .onAppear(() => {
            this.scrollToWeekInPicker(this.tempWeek);
          })
        }
        .layoutWeight(1)
        .height(140)
      }
      .width('75%')
      .height('40%')
      .padding({ left: 12, right: 12, top: 10, bottom: 8 })
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(12)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor($r('app.color.overlay'))
    .onClick(() => {
      this.showWeekPicker = false;
    })
  }

  private confirmWeekSelection(): void {
    this.currentWeek = this.tempWeek;
    this.showWeekPicker = false;
    this.currentWeekCourses = this.getCoursesForWeekOptimized(this.currentWeek);
    promptAction.showToast({ message: `已切换到第 ${this.currentWeek} 周` });
  }

  private showConflictDialog(course: Course, dayIndex: number, timeSlotGroup: TimeSlotData[]): void {
    const weekDayName = this.weekDays[dayIndex - 1];
    const dateStr = this.getFullDateForDay(dayIndex, this.currentWeek);
    const monthDayStr = dateStr.replace('月', '-').replace('日', '');
    
    const startSlot = course.timeSlot;
    const endSlot = course.timeSlot + course.duration - 1;
    const startTimeSlot = this.timeSlots.find(slot => slot.id === startSlot);
    const endTimeSlot = this.timeSlots.find(slot => slot.id === endSlot);
    
    let timeRange = '';
    let periodRange = '';
    if (startTimeSlot && endTimeSlot) {
      timeRange = `${startTimeSlot.time.split('-')[0]}-${endTimeSlot.time.split('-')[1]}`;
      periodRange = `${startTimeSlot.period}-${endTimeSlot.period}`;
    } else {
      timeRange = `${timeSlotGroup[0].time}-${timeSlotGroup[timeSlotGroup.length - 1].time}`;
      periodRange = `${timeSlotGroup[0].period}-${timeSlotGroup[timeSlotGroup.length - 1].period}`;
    }

    if (!course.conflictWith || course.conflictWith.length === 0) {
      promptAction.showDialog({
        title: '课程详情',
        message: `课程名称：${course.name}\n` +
          `授课教师：${course.teacher}\n` +
          `上课地点：${course.location}\n` +
          `上课时间：周${weekDayName} ${timeRange}（第${periodRange}节）\n` +
          `上课日期：${monthDayStr}\n` +
          `教学周次：第${course.week}周`,
        buttons: [
          { text: '确定', color: $r('app.color.primary') }
        ]
      });
      return;
    }

    const conflictCourses: Course[] = [course];
    course.conflictWith.forEach((conflictId: string) => {
      const conflictCourse = this.courses.find((c: Course) => c.id === conflictId);
      if (conflictCourse) {
        conflictCourses.push(conflictCourse);
      }
    });

    promptAction.showDialog({
      title: '课程冲突提示',
      message: `发现 ${conflictCourses.length} 门课程在同一时间上课\n` +
        conflictCourses.map((c, index) => {
          const cStartSlot = c.timeSlot;
          const cEndSlot = c.timeSlot + c.duration - 1;
          const cStartTimeSlot = this.timeSlots.find(slot => slot.id === cStartSlot);
          const cEndTimeSlot = this.timeSlots.find(slot => slot.id === cEndSlot);
          let cTimeRange = timeRange;
          let cPeriodRange = periodRange;
          if (cStartTimeSlot && cEndTimeSlot) {
            cTimeRange = `${cStartTimeSlot.time.split('-')[0]}-${cEndTimeSlot.time.split('-')[1]}`;
            cPeriodRange = `${cStartTimeSlot.period}-${cEndTimeSlot.period}`;
          }
          return `${index + 1}. ${c.name}（${c.location}）- ${c.teacher}\n` +
            `   时间：周${weekDayName} ${cTimeRange}（第${cPeriodRange}节）\n` +
            `   日期：${monthDayStr}`;
        }).join('\n\n'),
      buttons: [
        { text: '确定', color: $r('app.color.primary') }
      ]
    });
  }

  private getWeekArray(): number[] {
    const weeks: number[] = [];
    for (let i = 1; i <= this.settings.totalWeeks; i++) {
      weeks.push(i);
    }
    return weeks;
  }

  private getWeekDisplayText(week: number): string {
    return `${week}周`;
  }

  private isCurrentWeek(week: number): boolean {
    return week === this.currentWeek;
  }

  private scrollToWeekInPicker(week: number): void {
    const weeks = this.getWeekArray();
    const index = weeks.indexOf(week);
    if (index >= 0) {
      const rowHeight = 37;
      const rowsPerPage = 4;
      const targetOffset = Math.max(0, Math.floor(index / 4) * rowHeight - (rowsPerPage * rowHeight / 2));
      this.weekPickerScroller.scrollBy(0, targetOffset);
    }
  }

  private showCacheHistoryDialog(): void {
    this.loadCacheHistory();
    
    if (this.cachedDataList.length === 0) {
      promptAction.showDialog({
        title: '历史缓存数据',
        message: '暂无缓存数据',
        buttons: [
          { text: '确定', color: $r('app.color.primary') }
        ]
      });
      return;
    }

    const messageLines: string[] = [];
    
    for (let i = 0; i < this.cachedDataList.length; i++) {
      const data = this.cachedDataList[i];
      messageLines.push(`【版本 ${i + 1}】`);
      messageLines.push(`更新时间：${this.formatDateTime(data.lastUpdate)}`);
      messageLines.push(`课程数量：${data.courses.length} 节`);
      messageLines.push(`学期：${data.semester}`);
      if (i < this.cachedDataList.length - 1) {
        messageLines.push('');
      }
    }

    promptAction.showDialog({
      title: '历史缓存数据',
      message: messageLines.join('\n'),
      buttons: [
        { text: '确定', color: $r('app.color.primary') }
      ]
    });
  }

  private loadCacheHistory(): void {
    const v1 = this.localDataManager.getLatestScheduleData();
    const v2 = this.localDataManager.getPreviousScheduleData();
    const dataList: LocalScheduleData[] = [];
    if (v1) {
      dataList.push(v1);
    }
    if (v2) {
      dataList.push(v2);
    }
    this.cachedDataList = dataList;
  }

  private formatDateTime(isoString: string): string {
    try {
      const date = new Date(isoString);
      const year = date.getFullYear();
      const month = (date.getMonth() + 1).toString().padStart(2, '0');
      const day = date.getDate().toString().padStart(2, '0');
      const hours = date.getHours().toString().padStart(2, '0');
      const minutes = date.getMinutes().toString().padStart(2, '0');
      return `${year}年${month}月${day}日 ${hours}:${minutes}`;
    } catch (error) {
      return '未知时间';
    }
  }
}

export { SchedulePage };