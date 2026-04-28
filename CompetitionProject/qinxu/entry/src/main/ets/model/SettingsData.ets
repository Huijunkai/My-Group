// 课表设置数据模型
@Observed
export class SettingsData {
  // 学期信息（从数据库获取）
  semesterName: string = '';
  semesterStartDate: string = '';
  totalWeeks: number = 20;
  
  // 上课时间设置
  maxPeriods: number = 10;
  timeSlots: TimeSlot[] = [];
  
  // 显示设置
  showWeekend: boolean = true;
  showCurrentWeek: boolean = true;
  
  constructor() {
    this.timeSlots = this.getDefaultTimeSlots();
  }
  
  // 根据学期名称设置开学日期（已移除自动设置逻辑，由用户自定义）
  setSemesterStartDate(semester: string): void {
    // 不再自动设置开学日期，由用户在设置页面自定义
    return;
  }
  
  private getDefaultTimeSlots(): TimeSlot[] {
    return [
      new TimeSlot('08:30-09:15', '第1节'),
      new TimeSlot('09:20-10:05', '第2节'),
      new TimeSlot('10:25-11:10', '第3节'),
      new TimeSlot('11:15-12:00', '第4节'),
      new TimeSlot('14:30-15:15', '第5节'),
      new TimeSlot('15:20-16:05', '第6节'),
      new TimeSlot('16:15-17:00', '第7节'),
      new TimeSlot('17:05-17:50', '第8节'),
      new TimeSlot('18:20-19:05', '第9节'),
      new TimeSlot('19:10-19:55', '第10节'),
      new TimeSlot('20:05-20:50', '第11节'),
      new TimeSlot('20:55-21:40', '第12节')
    ];
  }
  
  /**
   * 计算当前周数
   * 根据开学日期和当前日期计算当前是第几周
   * 
   * @returns 当前周数（1到总周数之间）
   */
  getCurrentWeek(): number {
    if (!this.semesterStartDate) return 1;

    const today = new Date();
    const startDate = new Date(this.semesterStartDate);

    // 确保日期对象只包含年月日，忽略时分秒
    const todayOnlyDate = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const startOnlyDate = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate());

    // 计算两个日期之间的天数差
    const timeDiff = todayOnlyDate.getTime() - startOnlyDate.getTime();
    const dayDiff = Math.floor(timeDiff / (1000 * 3600 * 24));

    // 计算周数（从第1周开始）
    // 开学当天是第1周第1天
    // 第N周 = floor((天数差) / 7) + 1
    const currentWeek = Math.floor(dayDiff / 7) + 1;

    return Math.max(1, Math.min(currentWeek, this.totalWeeks));
  }
  
  // 验证时间槽是否有效
  validateTimeSlot(startTime: string, endTime: string): boolean {
    const start = this.parseTime(startTime);
    const end = this.parseTime(endTime);
    
    if (!start || !end) return false;
    if (end <= start) return false;
    
    // 检查是否与其他时间段冲突
    for (const slot of this.timeSlots) {
      const timeParts = slot.time.split('-');
      const existingStart = this.parseTime(timeParts[0]);
      const existingEnd = this.parseTime(timeParts[1]);
      
      if (existingStart && existingEnd) {
        if ((start >= existingStart && start < existingEnd) ||
            (end > existingStart && end <= existingEnd) ||
            (start <= existingStart && end >= existingEnd)) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  /**
   * 解析时间字符串为分钟数
   * 
   * @param timeStr - 时间字符串（HH:MM格式）
   * @returns 分钟数或null（格式无效时）
   */
  private parseTime(timeStr: string | null): number | null {
    if (!timeStr) return null;
    const match = timeStr.match(/^(\d{1,2}):(\d{2})$/);
    if (!match) return null;
    
    const hours = parseInt(match[1]);
    const minutes = parseInt(match[2]);
    
    if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) {
      return null;
    }
    
    return hours * 60 + minutes;
  }
}

/**
 * 上课时间段类
 * 
 * 表示一节课的时间信息
 * 
 * @example
 * ```typescript
 * const slot = new TimeSlot('08:30-09:15', '第1节');
 * ```
 */
@Observed
export class TimeSlot {
  /** 时间范围，如 "08:00-08:45" */
  time: string;
  /** 节次名称，如 "第1节" */
  period: string;
  
  constructor(time: string, period: string) {
    this.time = time;
    this.period = period;
  }
}

// 课程信息类 - 增强版
@Observed
export class CourseItem {
  id: string; // 课程唯一标识
  period: number; // 节次索引
  dayOfWeek: number; // 星期几（1-7，1=周一）
  name: string; // 课程名称
  location: string; // 上课地点
  teacher: string; // 授课教师
  weeks: number[]; // 上课周次
  color: string | Resource; // 课程颜色
  isActive: boolean = true; // 课程是否启用
  
  // 新增字段
  type: string = '专业课'; // 课程类型：专业课/选修课/实验课等
  startTime: string = ''; // 开始时间
  endTime: string = ''; // 结束时间
  building: string = ''; // 教学楼
  room: string = ''; // 教室号
  credit: number = 0; // 学分
  note: string = ''; // 备注
  
  constructor(
    period: number, 
    dayOfWeek: number, 
    name: string, 
    location: string, 
    teacher: string, 
    weeks: number[] = [], 
    color: string | Resource = '',
    id?: string,
    type: string = '专业课',
    startTime: string = '',
    endTime: string = '',
    building: string = '',
    room: string = '',
    credit: number = 0,
    note: string = ''
  ) {
    this.id = id || this.generateId();
    this.period = period;
    this.dayOfWeek = dayOfWeek;
    this.name = name;
    this.location = location;
    this.teacher = teacher;
    this.weeks = weeks;
    this.color = color || this.generateColor(name);
    this.type = type;
    this.startTime = startTime;
    this.endTime = endTime;
    this.building = building || this.extractBuilding(location);
    this.room = room || this.extractRoom(location);
    this.credit = credit;
    this.note = note;
  }
  
  /**
   * 生成课程唯一ID
   * 
   * @returns 唯一ID字符串
   */
  private generateId(): string {
    return `course_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * 根据课程名称生成颜色
   * 使用哈希算法从预设颜色中选择
   * 
   * @param courseName - 课程名称
   * @returns 十六进制颜色值
   */
  private generateColor(courseName: string): string {
    const colors = [
      '#4361ee', // 主蓝色
      '#3a0ca3', // 深紫色
      '#7209b7', // 紫色
      '#f72585', // 粉红色
      '#4cc9f0', // 浅蓝色
      '#4895ef', // 中蓝色
      '#4cc9f0', // 浅蓝色
      '#43aa8b', // 深绿色
      '#90be6d', // 浅绿色
      '#f9c74f', // 黄色
      '#f8961e', // 橙色
      '#f94144'  // 红色
    ];
    let hash = 0;
    for (let i = 0; i < courseName.length; i++) {
      hash = courseName.charCodeAt(i) + ((hash << 5) - hash);
    }
    return colors[Math.abs(hash) % colors.length];
  }
  
  /**
   * 从地点字符串中提取教学楼
   * 
   * @param location - 地点字符串
   * @returns 教学楼名称
   */
  private extractBuilding(location: string): string {
    const buildingMatch = location.match(/([A-Za-z\u4e00-\u9fa5]+楼|[A-Za-z\u4e00-\u9fa5]+馆)/);
    return buildingMatch ? buildingMatch[1] : '';
  }
  
  /**
   * 从地点字符串中提取教室号
   * 
   * @param location - 地点字符串
   * @returns 教室号
   */
  private extractRoom(location: string): string {
    const roomMatch = location.match(/(\d+-\d+|\d+[A-Za-z]?|[A-Za-z]\d+)/);
    return roomMatch ? roomMatch[1] : '';
  }
  
  /**
   * 获取完整的课程时间信息
   * 
   * @returns 时间范围字符串
   */
  getTimeInfo(): string {
    if (this.startTime && this.endTime) {
      return `${this.startTime}-${this.endTime}`;
    }
    return this.location;
  }
  
  /**
   * 获取详细的课程信息
   * 
   * @returns 包含教师、地点、周次、学分的详细信息字符串
   */
  getDetailInfo(): string {
    const parts: string[] = [];
    if (this.teacher) parts.push(`教师：${this.teacher}`);
    if (this.location) parts.push(`地点：${this.location}`);
    if (this.weeks.length > 0) parts.push(`周次：${this.weeks.join(',')}`);
    if (this.credit > 0) parts.push(`学分：${this.credit}`);
    return parts.join(' | ');
  }
  
  /**
   * 检查是否为当前课程（基于时间）
   * 
   * @returns 当前时间是否在课程时间范围内
   */
  isCurrentCourse(): boolean {
    if (!this.startTime || !this.endTime) return false;
    
    const now = new Date();
    const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    
    return currentTime >= this.startTime && currentTime <= this.endTime;
  }
  
  /**
   * 检查课程是否在当前周
   * 
   * @param currentWeek - 当前周数
   * @returns 是否在当前周
   */
  isInCurrentWeek(currentWeek: number): boolean {
    return this.weeks.length === 0 || this.weeks.includes(currentWeek);
  }
  
  /**
   * 检查课程是否与另一个课程冲突
   * 
   * @param otherCourse - 另一个课程
   * @returns 是否存在冲突
   */
  hasConflict(otherCourse: CourseItem): boolean {
    if (!this.isActive || !otherCourse.isActive) return false;
    if (this.dayOfWeek !== otherCourse.dayOfWeek) return false;
    if (this.period !== otherCourse.period) return false;
    
    if (this.weeks.length === 0 || otherCourse.weeks.length === 0) return true;
    
    return this.weeks.some(week => otherCourse.weeks.includes(week));
  }
  
  /**
   * 克隆课程
   * 
   * @returns 课程副本
   */
  clone(): CourseItem {
    return new CourseItem(
      this.period,
      this.dayOfWeek,
      this.name,
      this.location,
      this.teacher,
      [...this.weeks],
      this.color,
      this.id
    );
  }
}

/**
 * 课程管理器类
 * 
 * 提供课程的增删改查和冲突检测功能
 * 
 * @example
 * ```typescript
 * const manager = new CourseManager();
 * manager.addCourse(course);
 * const dayCourses = manager.getCoursesForDay(1);
 * ```
 */
@Observed
export class CourseManager {
  /** 课程列表 */
  courses: CourseItem[] = [];
  
  /**
   * 添加课程
   * 
   * @param course - 要添加的课程
   * @returns 是否添加成功（冲突时返回false）
   */
  addCourse(course: CourseItem): boolean {
    if (this.hasConflict(course)) {
      return false;
    }
    
    this.courses.push(course);
    return true;
  }
  
  /**
   * 更新课程
   * 
   * @param courseId - 课程ID
   * @param updatedCourse - 更新后的课程
   * @returns 是否更新成功
   */
  updateCourse(courseId: string, updatedCourse: CourseItem): boolean {
    const index = this.courses.findIndex(course => course.id === courseId);
    if (index === -1) return false;
    
    const tempCourses = this.courses.filter(course => course.id !== courseId);
    if (tempCourses.some(course => course.hasConflict(updatedCourse))) {
      return false;
    }
    
    this.courses[index] = updatedCourse;
    return true;
  }
  
  /**
   * 删除课程
   * 
   * @param courseId - 课程ID
   * @returns 是否删除成功
   */
  deleteCourse(courseId: string): boolean {
    const index = this.courses.findIndex(course => course.id === courseId);
    if (index === -1) return false;
    
    this.courses.splice(index, 1);
    return true;
  }
  
  /**
   * 根据ID获取课程
   * 
   * @param courseId - 课程ID
   * @returns 课程对象或undefined
   */
  getCourseById(courseId: string): CourseItem | undefined {
    return this.courses.find(course => course.id === courseId);
  }
  
  /**
   * 检查课程冲突
   * 
   * @param course - 要检查的课程
   * @returns 是否存在冲突
   */
  hasConflict(course: CourseItem): boolean {
    return this.courses.some(existingCourse => existingCourse.hasConflict(course));
  }
  
  /**
   * 获取某天某节的课程
   * 
   * @param dayOfWeek - 星期几
   * @param period - 节次
   * @returns 课程列表
   */
  getCoursesForPeriod(dayOfWeek: number, period: number): CourseItem[] {
    return this.courses.filter(course => 
      course.dayOfWeek === dayOfWeek && 
      course.period === period && 
      course.isActive
    );
  }
  
  /**
   * 获取某天的所有课程
   * 
   * @param dayOfWeek - 星期几
   * @returns 按节次排序的课程列表
   */
  getCoursesForDay(dayOfWeek: number): CourseItem[] {
    return this.courses.filter(course => 
      course.dayOfWeek === dayOfWeek && 
      course.isActive
    ).sort((a, b) => a.period - b.period);
  }
  
  /**
   * 搜索课程
   * 支持按名称、教师、地点搜索
   * 
   * @param keyword - 搜索关键词
   * @returns 匹配的课程列表
   */
  searchCourses(keyword: string): CourseItem[] {
    const lowerKeyword = keyword.toLowerCase();
    return this.courses.filter(course => 
      course.name.toLowerCase().includes(lowerKeyword) ||
      course.teacher.toLowerCase().includes(lowerKeyword) ||
      course.location.toLowerCase().includes(lowerKeyword)
    );
  }
  
  /**
   * 过滤课程（按周次）
   * 
   * @param week - 周次
   * @returns 该周的课程列表
   */
  filterCoursesByWeek(week: number): CourseItem[] {
    return this.courses.filter(course => course.isInCurrentWeek(week));
  }
  
  /**
   * 获取课程冲突列表
   * 返回所有存在冲突的课程组
   * 
   * @returns 冲突课程组列表
   */
  getConflictList(): CourseItem[][] {
    const conflicts: CourseItem[][] = [];
    const processed = new Set<string>();
    
    for (let i = 0; i < this.courses.length; i++) {
      const course1 = this.courses[i];
      if (processed.has(course1.id)) continue;
      
      const conflictGroup: CourseItem[] = [course1];
      
      for (let j = i + 1; j < this.courses.length; j++) {
        const course2 = this.courses[j];
        if (course1.hasConflict(course2)) {
          conflictGroup.push(course2);
          processed.add(course2.id);
        }
      }
      
      if (conflictGroup.length > 1) {
        conflicts.push(conflictGroup);
      }
      
      processed.add(course1.id);
    }
    
    return conflicts;
  }
}