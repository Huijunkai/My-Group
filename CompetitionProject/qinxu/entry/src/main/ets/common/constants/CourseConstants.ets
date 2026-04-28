/**
 * 课程常量配置模块
 * 
 * 定义课程相关的常量配置，包括：
 * - 课程时间安排
 * - 课程类型
 * - 课程颜色映射
 * - 默认时间段配置
 * 
 * @module CourseConstants
 */

/**
 * 课程常量类
 * 
 * 提供统一的课程配置管理
 * 
 * @example
 * ```typescript
 * const color = CourseConstants.getCourseColor('专业课');
 * // 结果: '#4a90d9'
 * ```
 */
export class CourseConstants {
  /** 每天最大节数 */
  static readonly MAX_PERIODS_PER_DAY: number = 12;
  
  /** 每周天数 */
  static readonly DAYS_PER_WEEK: number = 7;
  
  /** 默认总周数 */
  static readonly DEFAULT_TOTAL_WEEKS: number = 20;
  
  /**
   * 课程类型常量
   * 
   * @property MAJOR - 专业课
   * @property ELECTIVE - 选修课
   * @property LAB - 实验课
   * @property PE - 体育课
   * @property LECTURE - 讲座
   */
  static readonly COURSE_TYPES = {
    MAJOR: '专业课',
    ELECTIVE: '选修课',
    LAB: '实验课',
    PE: '体育课',
    LECTURE: '讲座'
  } as const;
  
  /**
   * 课程颜色映射
   * 
   * 为不同类型的课程分配不同的显示颜色
   * - 专业课: 蓝色 (#4a90d9)
   * - 选修课: 绿色 (#34c759)
   * - 实验课: 橙色 (#ff9500)
   * - 体育课: 红色 (#ff3b30)
   * - 讲座: 紫色 (#af52de)
   */
  static readonly COURSE_COLORS: Record<string, string> = {
    [CourseConstants.COURSE_TYPES.MAJOR]: '#4a90d9',
    [CourseConstants.COURSE_TYPES.ELECTIVE]: '#34c759',
    [CourseConstants.COURSE_TYPES.LAB]: '#ff9500',
    [CourseConstants.COURSE_TYPES.PE]: '#ff3b30',
    [CourseConstants.COURSE_TYPES.LECTURE]: '#af52de'
  };
  
  /**
   * 默认时间段配置
   * 
   * 定义每天 12 节课的时间安排：
   * - 上午: 第1-4节 (08:30-12:00)
   * - 下午: 第5-8节 (14:30-17:50)
   * - 晚上: 第9-12节 (18:20-21:40)
   */
  static readonly DEFAULT_TIME_SLOTS = [
    { time: '08:30-09:15', period: '第1节' },
    { time: '09:20-10:05', period: '第2节' },
    { time: '10:25-11:10', period: '第3节' },
    { time: '11:15-12:00', period: '第4节' },
    { time: '14:30-15:15', period: '第5节' },
    { time: '15:20-16:05', period: '第6节' },
    { time: '16:15-17:00', period: '第7节' },
    { time: '17:05-17:50', period: '第8节' },
    { time: '18:20-19:05', period: '第9节' },
    { time: '19:10-19:55', period: '第10节' },
    { time: '20:05-20:50', period: '第11节' },
    { time: '20:55-21:40', period: '第12节' }
  ] as const;
  
  /**
   * 获取课程颜色
   * 
   * 根据课程类型获取对应的显示颜色
   * 如果类型未定义，返回默认灰色
   * 
   * @param type - 课程类型
   * @returns 课程颜色（十六进制颜色值）
   */
  static getCourseColor(type: string): string {
    return CourseConstants.COURSE_COLORS[type] || '#8e8e93';
  }
}
