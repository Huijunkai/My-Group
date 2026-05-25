/**
 * 类型定义模块统一导出
 * 
 * 集中导出所有类型定义，便于统一引用
 * 
 * @module types
 * 
 * @example
 * ```typescript
 * import { ICourse, IGrade, IExam, IThemeConfig } from '../types';
 * ```
 */

/**
 * 课程信息接口
 */
export interface ICourse {
  /** 课程ID */
  id: string;
  /** 节次 */
  period: number;
  /** 星期几（1-7，1=周一） */
  dayOfWeek: number;
  /** 课程名称 */
  name: string;
  /** 上课地点 */
  location: string;
  /** 授课教师 */
  teacher: string;
  /** 上课周次数组 */
  weeks: number[];
  /** 课程颜色 */
  color: string | Resource;
  /** 是否启用 */
  isActive: boolean;
  /** 课程类型 */
  type: string;
  /** 开始时间 */
  startTime: string;
  /** 结束时间 */
  endTime: string;
  /** 教学楼 */
  building: string;
  /** 教室号 */
  room: string;
  /** 学分 */
  credit: number;
  /** 备注 */
  note: string;
}

/**
 * 时间段接口
 */
export interface ITimeSlot {
  /** 时间范围 */
  time: string;
  /** 节次名称 */
  period: string;
}

/**
 * 学期设置接口
 */
export interface ISemesterSettings {
  /** 学期名称 */
  semesterName: string;
  /** 开学日期 */
  semesterStartDate: string;
  /** 总周数 */
  totalWeeks: number;
  /** 每天最大节数 */
  maxPeriods: number;
  /** 时间段列表 */
  timeSlots: ITimeSlot[];
  /** 是否显示周末 */
  showWeekend: boolean;
  /** 是否显示当前周 */
  showCurrentWeek: boolean;
}

/**
 * 用户信息接口
 */
export interface IUserInfo {
  /** 学号 */
  studentId: string;
  /** 姓名 */
  name: string;
  /** 学院 */
  college: string;
  /** 专业 */
  major: string;
  /** 年级 */
  grade: string;
  /** 班级名称 */
  className: string;
}
