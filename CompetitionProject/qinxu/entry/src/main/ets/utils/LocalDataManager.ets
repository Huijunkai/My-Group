/**
 * 本地数据管理模块
 * 
 * 提供本地数据版本管理功能，包括：
 * - 课表数据版本控制
 * - 培养计划数据管理
 * - 成绩数据管理
 * - 数据版本切换
 * 
 * @module LocalDataManager
 */

import { preferences } from '@kit.ArkData';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'local_data_manager';

/**
 * 课表数据接口
 */
interface ScheduleData {
  /** 课程列表 */
  courses: CourseData[];
  /** 学期 */
  semester: string;
  /** 最后更新时间 */
  lastUpdate: string;
  /** 版本号 */
  version: number;
}

/**
 * 课程数据接口
 */
export interface CourseData {
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
  /** 节次 */
  period: string;
  /** 周次 */
  week: number;
  /** 课程类型 */
  courseType?: string;
}

/**
 * 培养计划数据接口
 */
interface TrainingPlanData {
  /** 课程列表 */
  courses: PlanCourseData[];
  /** 最后更新时间 */
  lastUpdate: string;
  /** 版本号 */
  version: number;
}

/**
 * 计划课程数据接口
 */
interface PlanCourseData {
  /** 课程名称 */
  courseName: string;
  /** 课程代码 */
  courseCode: string;
  /** 学分 */
  credit: number;
  /** 课程类型 */
  courseType: string;
  /** 状态 */
  status: string;
}

/**
 * 成绩数据接口
 */
interface GradeData {
  /** 成绩列表 */
  grades: GradeItem[];
  /** 最后更新时间 */
  lastUpdate: string;
  /** 版本号 */
  version: number;
  /** 新成绩数量 */
  newGradeCount: number;
}

/**
 * 成绩项接口
 */
interface GradeItem {
  /** 课程名称 */
  courseName: string;
  /** 课程代码 */
  courseCode: string;
  /** 学分 */
  credit: number;
  /** 成绩 */
  grade: string;
  /** 绩点 */
  gradePoint: string;
  /** 学期 */
  semester: string;
  /** 课程类型 */
  courseType: string;
  /** 是否新成绩 */
  isNew: boolean;
}

/**
 * 数据版本接口
 */
interface DataVersion {
  /** 课表版本1 */
  scheduleVersion1: ScheduleData | null;
  /** 课表版本2 */
  scheduleVersion2: ScheduleData | null;
  /** 培养计划版本1 */
  trainingPlanVersion1: TrainingPlanData | null;
  /** 培养计划版本2 */
  trainingPlanVersion2: TrainingPlanData | null;
  /** 成绩版本1 */
  gradeVersion1: GradeData | null;
  /** 成绩版本2 */
  gradeVersion2: GradeData | null;
}

/**
 * 本地数据管理类
 * 
 * 管理本地数据的版本控制和存储
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const manager = LocalDataManager.getInstance();
 * await manager.initialize(context);
 * const courses = manager.getCourses();
 * ```
 */
class LocalDataManager {
  /** 单例实例 */
  private static instance: LocalDataManager;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 数据版本 */
  private dataVersion: DataVersion = {
    scheduleVersion1: null,
    scheduleVersion2: null,
    trainingPlanVersion1: null,
    trainingPlanVersion2: null,
    gradeVersion1: null,
    gradeVersion2: null
  };

  static getInstance(): LocalDataManager {
    if (!LocalDataManager.instance) {
      LocalDataManager.instance = new LocalDataManager();
    }
    return LocalDataManager.instance;
  }

  async initialize(context: Context): Promise<void> {
    try {
      this.dataPreferences = await preferences.getPreferences(context, PREFERENCES_NAME);
      await this.loadAllData();
    } catch (error) {
      console.error('LocalDataManager: Failed to initialize:', error);
    }
  }

  private async loadAllData(): Promise<void> {
    if (!this.dataPreferences) return;

    try {
      const scheduleV1 = await this.dataPreferences.get('scheduleVersion1', '') as string;
      const scheduleV2 = await this.dataPreferences.get('scheduleVersion2', '') as string;
      const trainingPlanV1 = await this.dataPreferences.get('trainingPlanVersion1', '') as string;
      const trainingPlanV2 = await this.dataPreferences.get('trainingPlanVersion2', '') as string;
      const gradeV1 = await this.dataPreferences.get('gradeVersion1', '') as string;
      const gradeV2 = await this.dataPreferences.get('gradeVersion2', '') as string;

      if (scheduleV1) this.dataVersion.scheduleVersion1 = JSON.parse(scheduleV1) as ScheduleData;
      if (scheduleV2) this.dataVersion.scheduleVersion2 = JSON.parse(scheduleV2) as ScheduleData;
      if (trainingPlanV1) this.dataVersion.trainingPlanVersion1 = JSON.parse(trainingPlanV1) as TrainingPlanData;
      if (trainingPlanV2) this.dataVersion.trainingPlanVersion2 = JSON.parse(trainingPlanV2) as TrainingPlanData;
      if (gradeV1) this.dataVersion.gradeVersion1 = JSON.parse(gradeV1) as GradeData;
      if (gradeV2) this.dataVersion.gradeVersion2 = JSON.parse(gradeV2) as GradeData;
    } catch (error) {
      console.error('LocalDataManager: Failed to load data:', error);
    }
  }

  async saveScheduleData(courses: CourseData[], semester: string): Promise<boolean> {
    if (!this.dataPreferences) return false;

    const newSchedule: ScheduleData = {
      courses: courses,
      semester: semester,
      lastUpdate: new Date().toISOString(),
      version: Date.now()
    };

    const currentLatest = this.getLatestScheduleData();
    
    if (this.isScheduleDataChanged(currentLatest, newSchedule)) {
      this.dataVersion.scheduleVersion1 = this.dataVersion.scheduleVersion2;
      this.dataVersion.scheduleVersion2 = newSchedule;
      
      await this.dataPreferences.put('scheduleVersion1', 
        this.dataVersion.scheduleVersion1 ? JSON.stringify(this.dataVersion.scheduleVersion1) : '');
      await this.dataPreferences.put('scheduleVersion2', JSON.stringify(this.dataVersion.scheduleVersion2));
      await this.dataPreferences.flush();
      
      console.log('LocalDataManager: Schedule data saved with changes');
      return true;
    }

    console.log('LocalDataManager: Schedule data unchanged, skip saving');
    return false;
  }

  private isScheduleDataChanged(oldData: ScheduleData | null, newData: ScheduleData): boolean {
    if (!oldData) return true;
    if (oldData.semester !== newData.semester) return true;
    if (oldData.courses.length !== newData.courses.length) return true;

    const oldIds: string[] = [];
    for (let i = 0; i < oldData.courses.length; i++) {
      oldIds.push(oldData.courses[i].id);
    }

    const newIds: string[] = [];
    for (let i = 0; i < newData.courses.length; i++) {
      newIds.push(newData.courses[i].id);
    }

    oldIds.sort();
    newIds.sort();

    for (let i = 0; i < oldIds.length; i++) {
      if (oldIds[i] !== newIds[i]) return true;
    }

    return false;
  }

  getLatestScheduleData(): ScheduleData | null {
    if (this.dataVersion.scheduleVersion2) {
      return this.dataVersion.scheduleVersion2;
    }
    return this.dataVersion.scheduleVersion1;
  }

  getPreviousScheduleData(): ScheduleData | null {
    return this.dataVersion.scheduleVersion1;
  }

  async saveTrainingPlanData(courses: PlanCourseData[]): Promise<boolean> {
    if (!this.dataPreferences) return false;

    const newPlan: TrainingPlanData = {
      courses: courses,
      lastUpdate: new Date().toISOString(),
      version: Date.now()
    };

    const currentLatest = this.getLatestTrainingPlanData();
    
    if (this.isTrainingPlanDataChanged(currentLatest, newPlan)) {
      this.dataVersion.trainingPlanVersion1 = this.dataVersion.trainingPlanVersion2;
      this.dataVersion.trainingPlanVersion2 = newPlan;
      
      await this.dataPreferences.put('trainingPlanVersion1', 
        this.dataVersion.trainingPlanVersion1 ? JSON.stringify(this.dataVersion.trainingPlanVersion1) : '');
      await this.dataPreferences.put('trainingPlanVersion2', JSON.stringify(this.dataVersion.trainingPlanVersion2));
      await this.dataPreferences.flush();
      
      console.log('LocalDataManager: Training plan data saved with changes');
      return true;
    }

    console.log('LocalDataManager: Training plan data unchanged, skip saving');
    return false;
  }

  private isTrainingPlanDataChanged(oldData: TrainingPlanData | null, newData: TrainingPlanData): boolean {
    if (!oldData) return true;
    if (oldData.courses.length !== newData.courses.length) return true;

    const oldCodes: string[] = [];
    for (let i = 0; i < oldData.courses.length; i++) {
      oldCodes.push(oldData.courses[i].courseCode);
    }

    const newCodes: string[] = [];
    for (let i = 0; i < newData.courses.length; i++) {
      newCodes.push(newData.courses[i].courseCode);
    }

    oldCodes.sort();
    newCodes.sort();

    for (let i = 0; i < oldCodes.length; i++) {
      if (oldCodes[i] !== newCodes[i]) return true;
    }

    return false;
  }

  getLatestTrainingPlanData(): TrainingPlanData | null {
    if (this.dataVersion.trainingPlanVersion2) {
      return this.dataVersion.trainingPlanVersion2;
    }
    return this.dataVersion.trainingPlanVersion1;
  }

  async saveGradeData(grades: GradeItem[]): Promise<number> {
    if (!this.dataPreferences) return 0;

    const oldLatest = this.getLatestGradeData();
    const oldGradeMap: Map<string, GradeItem> = new Map();
    
    if (oldLatest) {
      for (let i = 0; i < oldLatest.grades.length; i++) {
        oldGradeMap.set(oldLatest.grades[i].courseCode, oldLatest.grades[i]);
      }
    }

    let newGradeCount = 0;
    const processedGrades: GradeItem[] = [];
    
    for (let i = 0; i < grades.length; i++) {
      const grade = grades[i];
      const oldGrade = oldGradeMap.get(grade.courseCode);
      
      if (!oldGrade) {
        grade.isNew = true;
        newGradeCount++;
      } else {
        grade.isNew = oldGrade.isNew;
      }
      processedGrades.push(grade);
    }

    const newGradeData: GradeData = {
      grades: processedGrades,
      lastUpdate: new Date().toISOString(),
      version: Date.now(),
      newGradeCount: newGradeCount
    };

    this.dataVersion.gradeVersion1 = this.dataVersion.gradeVersion2;
    this.dataVersion.gradeVersion2 = newGradeData;
    
    await this.dataPreferences.put('gradeVersion1', 
      this.dataVersion.gradeVersion1 ? JSON.stringify(this.dataVersion.gradeVersion1) : '');
    await this.dataPreferences.put('gradeVersion2', JSON.stringify(this.dataVersion.gradeVersion2));
    await this.dataPreferences.flush();
    
    console.log('LocalDataManager: Grade data saved, new grades:', newGradeCount);
    return newGradeCount;
  }

  getLatestGradeData(): GradeData | null {
    if (this.dataVersion.gradeVersion2) {
      return this.dataVersion.gradeVersion2;
    }
    return this.dataVersion.gradeVersion1;
  }

  clearGradeNewFlags(): void {
    if (this.dataVersion.gradeVersion2) {
      for (let i = 0; i < this.dataVersion.gradeVersion2.grades.length; i++) {
        this.dataVersion.gradeVersion2.grades[i].isNew = false;
      }
      this.dataVersion.gradeVersion2.newGradeCount = 0;
    }
  }

  isMidnightUpdateNeeded(): boolean {
    const latest = this.getLatestScheduleData();
    if (!latest) return true;

    const lastUpdate = new Date(latest.lastUpdate);
    const now = new Date();
    
    const lastUpdateDay = new Date(lastUpdate.getFullYear(), lastUpdate.getMonth(), lastUpdate.getDate());
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    
    return lastUpdateDay.getTime() < today.getTime();
  }

  getLastUpdateTime(): string {
    const latest = this.getLatestScheduleData();
    if (latest) {
      return latest.lastUpdate;
    }
    return '';
  }
}

export { 
  LocalDataManager, 
  ScheduleData, 
  TrainingPlanData, 
  PlanCourseData, 
  GradeData, 
  GradeItem 
};
