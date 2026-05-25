/**
 * 数据存储模块
 * 
 * 提供本地数据持久化功能，包括：
 * - 设置数据存储
 * - 课程数据存储
 * - 学生信息存储
 * - 成绩/考试/计划存储
 * - 数据导入导出
 * 
 * @module DataStore
 */

import preferences from '@ohos.data.preferences';
import common from '@ohos.app.ability.common';
import { SettingsData, TimeSlot, CourseItem } from '../model/SettingsData';

/**
 * 存储键名接口
 */
export interface StorageKeys {
  /** 设置键 */
  SETTINGS: string;
  /** 课程键 */
  COURSES: string;
  /** 自定义时间段键 */
  CUSTOM_TIME_SLOTS: string;
  /** 学生信息键 */
  STUDENT_INFO: string;
  /** 成绩键 */
  GRADES: string;
  /** 考试键 */
  EXAMS: string;
  /** 计划键 */
  PLANS: string;
  /** 进度键 */
  PROGRESS: string;
  /** 课表数据键 */
  SCHEDULE_DATA: string;
}

/**
 * 导出设置接口
 */
export interface ExportSettings {
  /** 学期名称 */
  semesterName: string;
  /** 开学日期 */
  semesterStartDate: string;
  /** 总周数 */
  totalWeeks: number;
  /** 最大节数 */
  maxPeriods: number;
  /** 显示周末 */
  showWeekend: boolean;
  /** 显示当前周 */
  showCurrentWeek: boolean;
}

/**
 * 时间段普通对象接口
 */
export interface TimeSlotPlain {
  /** 时间范围 */
  time: string;
  /** 节次名称 */
  period: string;
}

/**
 * 课程普通对象接口
 */
export interface CoursePlain {
  /** 星期几 */
  dayOfWeek?: string | number;
  /** 周次 */
  week?: number;
  /** 周次字符串 */
  weeks?: string;
  /** 节次 */
  period?: string;
  /** 课程名称 */
  name?: string;
  /** 上课地点 */
  location?: string;
  /** 授课教师 */
  teacher?: string;
  /** 学期 */
  semester?: string;
}

/**
 * 学生信息接口
 */
export interface StudentInfo {
  /** 学号 */
  studentId: string;
  /** 姓名 */
  name: string;
  /** 性别 */
  gender: string;
  /** 入学年份 */
  enrollmentYear: string;
  /** 班级名称 */
  className: string;
  /** 专业 */
  major: string;
  /** 学院 */
  college: string;
}

/**
 * 成绩接口
 */
export interface Grade {
  /** 学期 */
  semester: string;
  /** 课程代码 */
  courseCode: string;
  /** 课程名称 */
  courseName: string;
  /** 成绩 */
  score: string;
  /** 学分 */
  credit: string;
  /** 绩点 */
  gradePoint: string;
  /** 课程类型 */
  courseType: string;
  /** 考试类型 */
  examType: string;
}

/**
 * 考试接口
 */
export interface Exam {
  /** 课程名称 */
  courseName: string;
  /** 考试时间 */
  examTime: string;
  /** 考试地点 */
  location: string;
  /** 座位号 */
  seatNumber: string;
  /** 考试类型 */
  examType: string;
  /** 状态 */
  status: string;
}

/**
 * 计划接口
 */
export interface Plan {
  /** 学期 */
  semester: string;
  /** 课程代码 */
  courseCode: string;
  /** 课程名称 */
  courseName: string;
  /** 学分 */
  credit: string;
  /** 总学时 */
  totalHours: string;
  /** 课程类型 */
  courseType: string;
  /** 课程属性 */
  courseAttribute: string;
  /** 考试类型 */
  examType: string;
}

/**
 * 进度接口
 */
export interface Progress {
  /** 类别 */
  category: string;
  /** 要求学分 */
  requiredCredits: string;
  /** 已完成学分 */
  completedCredits: string;
  /** 当前学分 */
  currentCredits: string;
  /** 剩余学分 */
  remainingCredits: string;
}

/**
 * 课表数据接口
 */
export interface ScheduleData {
  /** 课程列表 */
  courses: Object[];
  /** 学期 */
  semester: string;
  /** 最后更新时间 */
  lastUpdated: number;
}

/**
 * 导出数据接口
 */
export interface ExportData {
  /** 设置 */
  settings?: ExportSettings;
  courses?: CoursePlain[];
  timeSlots?: TimeSlotPlain[];
}

interface DialogButton {
  text: string;
  color: string;
}

export class DataStore {
  private static instance: DataStore;
  private preferences: preferences.Preferences | null = null;

  private static readonly KEYS: StorageKeys = {
    SETTINGS: 'course_settings',
    COURSES: 'course_data',
    CUSTOM_TIME_SLOTS: 'custom_time_slots',
    STUDENT_INFO: 'student_info',
    GRADES: 'grades_data',
    EXAMS: 'exams_data',
    PLANS: 'plans_data',
    PROGRESS: 'progress_data',
    SCHEDULE_DATA: 'schedule_data'
  };

  private constructor() {}

  public static getInstance(): DataStore {
    if (!DataStore.instance) {
      DataStore.instance = new DataStore();
    }
    return DataStore.instance;
  }

  public async initialize(context: common.Context): Promise<void> {
    try {
      this.preferences = await preferences.getPreferences(context, 'course_schedule_db');
      console.log('DataStore initialized successfully');
    } catch (error) {
      console.error('Failed to initialize DataStore:', error);
    }
  }

  public async saveSettings(settings: SettingsData): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const settingsJson = JSON.stringify({
        semesterName: settings.semesterName,
        semesterStartDate: settings.semesterStartDate,
        totalWeeks: settings.totalWeeks,
        maxPeriods: settings.maxPeriods,
        showWeekend: settings.showWeekend,
        showCurrentWeek: settings.showCurrentWeek
      });

      console.log('DataStore: saveSettings - settingsJson:', settingsJson);

      await this.preferences.put(DataStore.KEYS.SETTINGS, settingsJson);
      await this.preferences.flush();
      
      console.log('DataStore: saveSettings - success');
      
      return true;
    } catch (error) {
      console.error('Failed to save settings:', error);
      return false;
    }
  }

  public async loadSettings(): Promise<SettingsData | null> {
    if (!this.preferences) return null;

    try {
      const settingsJson = await this.preferences.get(DataStore.KEYS.SETTINGS, '{}') as string;
      
      console.log('DataStore: loadSettings - settingsJson:', settingsJson);
      
      const settingsData: Partial<ExportSettings> = JSON.parse(settingsJson) as Partial<ExportSettings>;

      const settings = new SettingsData();
      
      if (settingsData.semesterName !== undefined && settingsData.semesterName !== '') {
        settings.semesterName = settingsData.semesterName;
      }
      if (settingsData.semesterStartDate !== undefined && settingsData.semesterStartDate !== '') {
        settings.semesterStartDate = settingsData.semesterStartDate;
      }
      if (settingsData.totalWeeks !== undefined && settingsData.totalWeeks > 0) {
        settings.totalWeeks = settingsData.totalWeeks;
      }
      if (settingsData.maxPeriods !== undefined && settingsData.maxPeriods > 0) {
        settings.maxPeriods = settingsData.maxPeriods;
      }
      if (settingsData.showWeekend !== undefined) {
        settings.showWeekend = settingsData.showWeekend;
      }
      if (settingsData.showCurrentWeek !== undefined) {
        settings.showCurrentWeek = settingsData.showCurrentWeek;
      }

      console.log('DataStore: loadSettings - loaded settings:', {
        semesterName: settings.semesterName,
        semesterStartDate: settings.semesterStartDate,
        totalWeeks: settings.totalWeeks
      });

      return settings;
    } catch (error) {
      console.error('Failed to load settings:', error);
      return null;
    }
  }

  public async saveTimeSlots(timeSlots: TimeSlot[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const plainSlots: TimeSlotPlain[] = timeSlots.map<TimeSlotPlain>((slot: TimeSlot): TimeSlotPlain => ({
        time: slot.time,
        period: slot.period
      }));

      const timeSlotsJson = JSON.stringify(plainSlots);

      await this.preferences.put(DataStore.KEYS.CUSTOM_TIME_SLOTS, timeSlotsJson);
      await this.preferences.flush();
      return true;
    } catch (error) {
      console.error('Failed to save time slots:', error);
      return false;
    }
  }

  public async loadTimeSlots(): Promise<TimeSlot[]> {
    if (!this.preferences) return [];

    try {
      const timeSlotsJson = await this.preferences.get(DataStore.KEYS.CUSTOM_TIME_SLOTS, '[]') as string;
      const timeSlotsData: TimeSlotPlain[] = JSON.parse(timeSlotsJson) as TimeSlotPlain[];

      return timeSlotsData.map((slotData: TimeSlotPlain) =>
        new TimeSlot(slotData.time, slotData.period)
      );
    } catch (error) {
      console.error('Failed to load time slots:', error);
      return [];
    }
  }

  public async saveCourses(courses: CourseItem[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const plainCourses: CoursePlain[] = [];
      for (let i = 0; i < courses.length; i++) {
        const course = courses[i];
        const plain: CoursePlain = {
          period: String(course.period),
          dayOfWeek: course.dayOfWeek,
          name: course.name,
          location: course.location,
          teacher: course.teacher,
          weeks: course.weeks ? course.weeks.join(',') : ''
        };
        plainCourses.push(plain);
      }

      const coursesJson = JSON.stringify(plainCourses);

      await this.preferences.put(DataStore.KEYS.COURSES, coursesJson);
      await this.preferences.flush();
      return true;
    } catch (error) {
      console.error('Failed to save courses:', error);
      return false;
    }
  }

  public async loadCourses(): Promise<CourseItem[]> {
    if (!this.preferences) return [];

    try {
      const coursesJson = await this.preferences.get(DataStore.KEYS.COURSES, '[]') as string;
      const coursesData: CoursePlain[] = JSON.parse(coursesJson) as CoursePlain[];

      const result: CourseItem[] = [];
      for (let i = 0; i < coursesData.length; i++) {
        const courseData = coursesData[i];
        const dayOfWeek = courseData.dayOfWeek ? Number(courseData.dayOfWeek) : 1;
        const period = courseData.period ? Number(courseData.period) : 1;
        const weeks: number[] = [];
        if (courseData.weeks) {
          const weeksStr = String(courseData.weeks);
          const parts = weeksStr.split(',');
          for (let j = 0; j < parts.length; j++) {
            const w = parseInt(parts[j].trim());
            if (!isNaN(w)) {
              weeks.push(w);
            }
          }
        }
        const course = new CourseItem(
          period,
          dayOfWeek,
          courseData.name ? courseData.name : '',
          courseData.location ? courseData.location : '',
          courseData.teacher ? courseData.teacher : '',
          weeks,
          ''
        );
        result.push(course);
      }
      return result;
    } catch (error) {
      console.error('Failed to load courses:', error);
      return [];
    }
  }

  public async saveStudentInfo(info: StudentInfo): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const infoJson = JSON.stringify(info);
      await this.preferences.put(DataStore.KEYS.STUDENT_INFO, infoJson);
      await this.preferences.flush();
      console.log('DataStore: Student info saved');
      return true;
    } catch (error) {
      console.error('Failed to save student info:', error);
      return false;
    }
  }

  public async loadStudentInfo(): Promise<StudentInfo | null> {
    if (!this.preferences) return null;

    try {
      const infoJson = await this.preferences.get(DataStore.KEYS.STUDENT_INFO, '') as string;
      if (!infoJson) return null;
      const info: StudentInfo = JSON.parse(infoJson) as StudentInfo;
      console.log('DataStore: Student info loaded');
      return info;
    } catch (error) {
      console.error('Failed to load student info:', error);
      return null;
    }
  }

  public async saveGrades(grades: Grade[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const gradesJson = JSON.stringify(grades);
      await this.preferences.put(DataStore.KEYS.GRADES, gradesJson);
      await this.preferences.flush();
      console.log('DataStore: Grades saved');
      return true;
    } catch (error) {
      console.error('Failed to save grades:', error);
      return false;
    }
  }

  public async loadGrades(): Promise<Grade[]> {
    if (!this.preferences) return [];

    try {
      const gradesJson = await this.preferences.get(DataStore.KEYS.GRADES, '[]') as string;
      const grades: Grade[] = JSON.parse(gradesJson) as Grade[];
      console.log('DataStore: Grades loaded');
      return grades;
    } catch (error) {
      console.error('Failed to load grades:', error);
      return [];
    }
  }

  public async saveExams(exams: Exam[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const examsJson = JSON.stringify(exams);
      await this.preferences.put(DataStore.KEYS.EXAMS, examsJson);
      await this.preferences.flush();
      console.log('DataStore: Exams saved');
      return true;
    } catch (error) {
      console.error('Failed to save exams:', error);
      return false;
    }
  }

  public async loadExams(): Promise<Exam[]> {
    if (!this.preferences) return [];

    try {
      const examsJson = await this.preferences.get(DataStore.KEYS.EXAMS, '[]') as string;
      const exams: Exam[] = JSON.parse(examsJson) as Exam[];
      console.log('DataStore: Exams loaded');
      return exams;
    } catch (error) {
      console.error('Failed to load exams:', error);
      return [];
    }
  }

  public async savePlans(plans: Plan[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const plansJson = JSON.stringify(plans);
      await this.preferences.put(DataStore.KEYS.PLANS, plansJson);
      await this.preferences.flush();
      console.log('DataStore: Plans saved');
      return true;
    } catch (error) {
      console.error('Failed to save plans:', error);
      return false;
    }
  }

  public async loadPlans(): Promise<Plan[]> {
    if (!this.preferences) return [];

    try {
      const plansJson = await this.preferences.get(DataStore.KEYS.PLANS, '[]') as string;
      const plans: Plan[] = JSON.parse(plansJson) as Plan[];
      console.log('DataStore: Plans loaded');
      return plans;
    } catch (error) {
      console.error('Failed to load plans:', error);
      return [];
    }
  }

  public async saveProgress(progress: Progress[]): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const progressJson = JSON.stringify(progress);
      await this.preferences.put(DataStore.KEYS.PROGRESS, progressJson);
      await this.preferences.flush();
      console.log('DataStore: Progress saved');
      return true;
    } catch (error) {
      console.error('Failed to save progress:', error);
      return false;
    }
  }

  public async loadProgress(): Promise<Progress[]> {
    if (!this.preferences) return [];

    try {
      const progressJson = await this.preferences.get(DataStore.KEYS.PROGRESS, '[]') as string;
      const progress: Progress[] = JSON.parse(progressJson) as Progress[];
      console.log('DataStore: Progress loaded');
      return progress;
    } catch (error) {
      console.error('Failed to load progress:', error);
      return [];
    }
  }

  public async saveScheduleData(data: ScheduleData): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      const dataJson = JSON.stringify(data);
      await this.preferences.put(DataStore.KEYS.SCHEDULE_DATA, dataJson);
      await this.preferences.flush();
      console.log('DataStore: Schedule data saved');
      return true;
    } catch (error) {
      console.error('Failed to save schedule data:', error);
      return false;
    }
  }

  public async loadScheduleData(): Promise<ScheduleData | null> {
    if (!this.preferences) return null;

    try {
      const dataJson = await this.preferences.get(DataStore.KEYS.SCHEDULE_DATA, '') as string;
      if (!dataJson) return null;
      const data: ScheduleData = JSON.parse(dataJson) as ScheduleData;
      console.log('DataStore: Schedule data loaded');
      return data;
    } catch (error) {
      console.error('Failed to load schedule data:', error);
      return null;
    }
  }

  public async clearAllData(): Promise<boolean> {
    if (!this.preferences) return false;

    try {
      await this.preferences.delete(DataStore.KEYS.SETTINGS);
      await this.preferences.delete(DataStore.KEYS.COURSES);
      await this.preferences.delete(DataStore.KEYS.CUSTOM_TIME_SLOTS);
      await this.preferences.delete(DataStore.KEYS.STUDENT_INFO);
      await this.preferences.delete(DataStore.KEYS.GRADES);
      await this.preferences.delete(DataStore.KEYS.EXAMS);
      await this.preferences.delete(DataStore.KEYS.PLANS);
      await this.preferences.delete(DataStore.KEYS.PROGRESS);
      await this.preferences.delete(DataStore.KEYS.SCHEDULE_DATA);
      await this.preferences.flush();
      return true;
    } catch (error) {
      console.error('Failed to clear data:', error);
      return false;
    }
  }

  public async exportData(): Promise<string> {
    const settings = await this.loadSettings();
    const courses = await this.loadCourses();
    const timeSlots = await this.loadTimeSlots();

    const exportData: ExportData = {
      settings: settings ? {
        semesterName: settings.semesterName,
        semesterStartDate: settings.semesterStartDate,
        totalWeeks: settings.totalWeeks,
        maxPeriods: settings.maxPeriods,
        showWeekend: settings.showWeekend,
        showCurrentWeek: settings.showCurrentWeek
      } : undefined,
      courses: [],
      timeSlots: []
    };

    for (let i = 0; i < courses.length; i++) {
      const course = courses[i];
      const plain: CoursePlain = {
        period: String(course.period),
        dayOfWeek: course.dayOfWeek,
        name: course.name,
        location: course.location,
        teacher: course.teacher,
        weeks: course.weeks ? course.weeks.join(',') : ''
      };
      exportData.courses!.push(plain);
    }

    for (let i = 0; i < timeSlots.length; i++) {
      const slot = timeSlots[i];
      const plain: TimeSlotPlain = {
        time: slot.time,
        period: slot.period
      };
      exportData.timeSlots!.push(plain);
    }

    return JSON.stringify(exportData, null, 2);
  }

  public async importData(dataJson: string): Promise<boolean> {
    try {
      const data: ExportData = JSON.parse(dataJson) as ExportData;

      if (data.settings) {
        const settings = new SettingsData();
        settings.semesterName = data.settings.semesterName;
        settings.semesterStartDate = data.settings.semesterStartDate;
        settings.totalWeeks = data.settings.totalWeeks;
        settings.maxPeriods = data.settings.maxPeriods;
        settings.showWeekend = data.settings.showWeekend;
        settings.showCurrentWeek = data.settings.showCurrentWeek;
        await this.saveSettings(settings);
      }

      if (data.courses) {
        const courses: CourseItem[] = [];
        for (let i = 0; i < data.courses.length; i++) {
          const courseData = data.courses[i];
          const dayOfWeek = courseData.dayOfWeek ? Number(courseData.dayOfWeek) : 1;
          const period = courseData.period ? Number(courseData.period) : 1;
          const weeks: number[] = [];
          if (courseData.weeks) {
            const weeksStr = String(courseData.weeks);
            const parts = weeksStr.split(',');
            for (let j = 0; j < parts.length; j++) {
              const w = parseInt(parts[j].trim());
              if (!isNaN(w)) {
                weeks.push(w);
              }
            }
          }
          const course = new CourseItem(
            period,
            dayOfWeek,
            courseData.name ? courseData.name : '',
            courseData.location ? courseData.location : '',
            courseData.teacher ? courseData.teacher : '',
            weeks,
            ''
          );
          courses.push(course);
        }
        await this.saveCourses(courses);
      }
      
      if (data.timeSlots) {
        const timeSlots = data.timeSlots.map((slotData: TimeSlotPlain) =>
          new TimeSlot(slotData.time, slotData.period));
        await this.saveTimeSlots(timeSlots);
      }
      
      return true;
    } catch (error) {
      console.error('Failed to import data:', error);
      return false;
    }
  }
}
