/**
 * 课表数据源模块
 * 
 * 提供课表数据缓存和快速查询功能，包括：
 * - 课表数据缓存
 * - 课程索引构建
 * - O(1)时间复杂度查询
 * 
 * @module ScheduleDataSource
 */

import { DataStore } from './DataStore';
import preferences from '@ohos.data.preferences';
import { common } from '@kit.AbilityKit';

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
}

/**
 * 课表数据接口
 */
interface ScheduleData {
  /** 课程列表 */
  courses: Course[];
  /** 学期 */
  semester: string;
  /** 最后更新时间 */
  lastUpdated: number;
  /** 预构建的索引结构 */
  courseIndex?: CourseIndex;
}

/**
 * 课程索引结构接口
 * 用于O(1)时间复杂度查询
 */
interface CourseIndex {
  /** 按周索引: week -> 课程列表 */
  byWeek: Map<number, Course[]>;
  /** 按周+星期索引: "week-day" -> 课程列表 */
  byWeekDay: Map<string, Course[]>;
  /** 按周+星期+时间段索引: "week-day-slot" -> 课程 */
  byWeekDaySlot: Map<string, Course>;
  /** 课程总数 */
  totalCount: number;
}

/**
 * 加载选项接口
 */
interface LoadOptions {
  /** 是否强制刷新 */
  forceRefresh?: boolean;
  /** 学期 */
  semester?: string;
}

/**
 * 节次范围接口
 */
interface PeriodRange {
  /** 开始节次 */
  startSlot: number;
  /** 持续节数 */
  duration: number;
}

/**
 * 生成的对象字面量接口
 */
interface GeneratedObjectLiteralInterface_1 {
  /** 课程列表 */
  courses: Course[];
  /** 学期 */
  semester: string;
  /** 最后更新时间 */
  lastUpdated: number;
}

/**
 * 课表数据源类
 * 
 * 提供课表数据的缓存和快速查询功能
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const dataSource = ScheduleDataSource.getInstance();
 * await dataSource.initialize(context);
 * const schedule = await dataSource.loadSchedule(studentId);
 * ```
 */
export class ScheduleDataSource {
  /** 单例实例 */
  private static instance: ScheduleDataSource;
  /** 缓存Map */
  private cache: Map<string, ScheduleData> = new Map();
  /** 偏好设置对象 */
  private preferences: preferences.Preferences | null = null;
  /** 缓存有效期（30分钟） */
  private readonly CACHE_DURATION = 30 * 60 * 1000;
  /** 内存中的快速查询缓存 */
  private queryCache: Map<string, Course[]> = new Map();

  private constructor() {}

  public static getInstance(): ScheduleDataSource {
    if (!ScheduleDataSource.instance) {
      ScheduleDataSource.instance = new ScheduleDataSource();
    }
    return ScheduleDataSource.instance;
  }

  public async initialize(context: common.Context): Promise<void> {
    try {
      this.preferences = await preferences.getPreferences(context, 'schedule_cache_db');
      console.log('ScheduleDataSource initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ScheduleDataSource:', error);
    }
  }

  public async loadSchedule(studentId: string, options: LoadOptions = {}): Promise<ScheduleData> {
    const forceRefresh = options.forceRefresh !== undefined ? options.forceRefresh : false;
    const semester = options.semester;
    const cacheKey = this.getCacheKey(studentId, semester);

    if (!forceRefresh) {
      const cachedData = await this.getCachedSchedule(cacheKey);
      if (cachedData && this.isCacheValid(cachedData)) {
        console.log('ScheduleDataSource: Using cached data, semester:', cachedData.semester);
        // 确保索引存在
        if (!cachedData.courseIndex) {
          cachedData.courseIndex = this.buildCourseIndex(cachedData.courses);
        }
        return cachedData;
      }
    }

    console.log('ScheduleDataSource: Loading fresh data');
    const freshData = await this.loadFromNetwork(studentId, semester);
    // 构建索引
    freshData.courseIndex = this.buildCourseIndex(freshData.courses);
    await this.cacheSchedule(cacheKey, freshData);
    return freshData;
  }

  public async refreshSchedule(studentId: string, semester?: string): Promise<ScheduleData> {
    return this.loadSchedule(studentId, { forceRefresh: true, semester });
  }

  public async clearCache(studentId: string, semester?: string): Promise<void> {
    const cacheKey = this.getCacheKey(studentId, semester);
    this.cache.delete(cacheKey);
    this.queryCache.clear();

    if (this.preferences) {
      try {
        await this.preferences.delete(cacheKey);
        await this.preferences.flush();
      } catch (error) {
        console.error('Failed to clear cache:', error);
      }
    }
  }

  public async clearAllCache(): Promise<void> {
    this.cache.clear();
    this.queryCache.clear();

    if (this.preferences) {
      try {
        await this.preferences.clear();
        await this.preferences.flush();
      } catch (error) {
        console.error('Failed to clear all cache:', error);
      }
    }
  }

  // 新增：快速查询方法 - 按周获取课程
  public getCoursesByWeek(data: ScheduleData, week: number): Course[] {
    if (data.courseIndex?.byWeek.has(week)) {
      return data.courseIndex.byWeek.get(week) || [];
    }
    return [];
  }

  // 新增：快速查询方法 - 按周和星期获取课程
  public getCoursesByWeekDay(data: ScheduleData, week: number, dayOfWeek: number): Course[] {
    const key = `${week}-${dayOfWeek}`;
    if (data.courseIndex?.byWeekDay.has(key)) {
      return data.courseIndex.byWeekDay.get(key) || [];
    }
    return [];
  }

  // 新增：快速查询方法 - 获取特定时间段的课程
  public getCourseByWeekDaySlot(data: ScheduleData, week: number, dayOfWeek: number, slot: number): Course | null {
    const key = `${week}-${dayOfWeek}-${slot}`;
    if (data.courseIndex?.byWeekDaySlot.has(key)) {
      return data.courseIndex.byWeekDaySlot.get(key) || null;
    }
    return null;
  }

  // 新增：构建课程索引
  private buildCourseIndex(courses: Course[]): CourseIndex {
    const byWeek = new Map<number, Course[]>();
    const byWeekDay = new Map<string, Course[]>();
    const byWeekDaySlot = new Map<string, Course>();

    for (let i = 0; i < courses.length; i++) {
      const course = courses[i];

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

    return {
      byWeek,
      byWeekDay,
      byWeekDaySlot,
      totalCount: courses.length
    };
  }

  private async loadFromNetwork(studentId: string, semester?: string): Promise<ScheduleData> {
    try {
      const dataStore = DataStore.getInstance();
      const scheduleData = await dataStore.loadScheduleData();
      
      if (scheduleData && scheduleData.courses) {
        const backendCourses: Record<string, Object>[] = scheduleData.courses as Record<string, Object>[];
        const courses = this.mapBackendCourses(backendCourses);
        const result: ScheduleData = {
          courses: courses,
          semester: scheduleData.semester || semester || '',
          lastUpdated: scheduleData.lastUpdated || Date.now()
        };
        return result;
      }

      const emptyData: ScheduleData = {
        courses: [],
        semester: semester || '',
        lastUpdated: Date.now()
      };
      return emptyData;
    } catch (error) {
      console.error('Failed to load from local storage:', error);
      throw new Error('Failed to load schedule data');
    }
  }

  private mapBackendCourses(items: Record<string, Object>[]): Course[] {
    const courses: Course[] = [];
    const colors: Array<string | Resource> = [
      '#4A90E2', '#5AC8FA', $r('app.color.success'), '#FF9500', '#FF3B30',
      '#AF52DE', '#5856D6', '#FF2D55', '#FFCC00', '#007AFF',
      '#00C7BE', '#8E8E93', '#FF6B9D', '#C9A227', '#6B4C9A'
    ];

    const dayMap: Record<string, number> = {
      '星期一': 1, '星期二': 2, '星期三': 3, '星期四': 4,
      '星期五': 5, '星期六': 6, '星期日': 7,
      '周一': 1, '周二': 2, '周三': 3, '周四': 4,
      '周五': 5, '周六': 6, '周日': 7
    };

    const courseColorMap: Map<string, string | Resource> = new Map();

    // 预分配数组容量，避免动态扩容
    const estimatedSize = items.length;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const dayOfWeek = this.parseDayOfWeek(item['dayOfWeek'], dayMap);
      const periodStr = String(item['period'] || '1');
      const periodRange = this.parsePeriodRange(periodStr);
      const name = String(item['name'] || '');
      const teacher = String(item['teacher'] || '');
      const location = String(item['location'] || '');
      const week = Number(item['week'] || 1);

      let color: string | Resource;
      if (courseColorMap.has(name)) {
        color = courseColorMap.get(name)!;
      } else {
        const hash = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        const colorIndex = hash % colors.length;
        color = colors[colorIndex];
        courseColorMap.set(name, color);
      }

      const course: Course = {
        id: `${name}_${dayOfWeek}_${week}_${periodRange.startSlot}`,
        name: name,
        teacher: teacher,
        location: location,
        dayOfWeek: dayOfWeek,
        timeSlot: periodRange.startSlot,
        duration: periodRange.duration,
        week: week,
        color: color
      };
      courses.push(course);
    }

    return courses;
  }

  private parseDayOfWeek(value: Object | undefined, dayMap: Record<string, number>): number {
    if (value === undefined || value === null) return 1;

    const num = Number(value);
    if (!Number.isNaN(num) && num >= 1 && num <= 7) {
      return num;
    }

    const str = String(value).trim();
    if (dayMap[str]) {
      return dayMap[str];
    }

    return 1;
  }

  private parsePeriodRange(periodStr: string): PeriodRange {
    const defaultRange: PeriodRange = { startSlot: 1, duration: 2 };
    if (!periodStr) return defaultRange;

    const trimmed = periodStr.trim();

    const rangeMatch = trimmed.match(/(\d{1,2})\s*[-~]\s*(\d{1,2})/);
    if (rangeMatch) {
      const start = parseInt(rangeMatch[1], 10);
      const end = parseInt(rangeMatch[2], 10);
      const result: PeriodRange = { startSlot: start, duration: end - start + 1 };
      return result;
    }

    const num = Number(trimmed);
    if (!Number.isNaN(num)) {
      const result: PeriodRange = { startSlot: num, duration: 2 };
      return result;
    }

    const singleMatch = trimmed.match(/^(\d+)/);
    if (singleMatch) {
      const result: PeriodRange = { startSlot: parseInt(singleMatch[1], 10), duration: 2 };
      return result;
    }

    return defaultRange;
  }

  private async getCachedSchedule(cacheKey: string): Promise<ScheduleData | null> {
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey) || null;
    }

    if (this.preferences) {
      try {
        const cachedString = await this.preferences.get(cacheKey, '');
        if (cachedString) {
          const cachedData = JSON.parse(cachedString as string) as ScheduleData;
          // 重建索引（因为JSON序列化会丢失Map结构）
          cachedData.courseIndex = this.buildCourseIndex(cachedData.courses);
          this.cache.set(cacheKey, cachedData);
          return cachedData;
        }
      } catch (error) {
        console.error('Failed to get cached data:', error);
      }
    }

    return null;
  }

  private async cacheSchedule(cacheKey: string, data: ScheduleData): Promise<void> {
    this.cache.set(cacheKey, data);

    if (this.preferences) {
      try {
        // 序列化时不包含索引，因为Map无法被JSON正确序列化
        const dataToCache: GeneratedObjectLiteralInterface_1 = {
          courses: data.courses,
          semester: data.semester,
          lastUpdated: data.lastUpdated
        };
        const dataString = JSON.stringify(dataToCache);
        await this.preferences.put(cacheKey, dataString);
        await this.preferences.flush();
      } catch (error) {
        console.error('Failed to cache data:', error);
      }
    }
  }

  private isCacheValid(data: ScheduleData): boolean {
    const now = Date.now();
    const age = now - data.lastUpdated;
    return age < this.CACHE_DURATION;
  }

  private getCacheKey(studentId: string, semester?: string): string {
    return semester ? `schedule_${studentId}_${semester}` : `schedule_${studentId}`;
  }
}
