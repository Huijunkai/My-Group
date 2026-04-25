/**
 * 学业相关类型定义模块
 * 
 * 定义学业相关的数据类型，包括：
 * - 成绩数据类型
 * - 考试数据类型
 * - 学分进度类型
 * 
 * @module academic
 */

/**
 * 成绩信息接口
 */
export interface IGrade {
  /** 成绩ID */
  id: string;
  /** 课程名称 */
  courseName: string;
  /** 课程类型 */
  courseType: string;
  /** 学分 */
  credit: number;
  /** 成绩分数 */
  score: string;
  /** 绩点 */
  gradePoint: number;
  /** 学期 */
  semester: string;
  /** 授课教师 */
  teacher: string;
}

/**
 * 考试信息接口
 */
export interface IExam {
  /** 考试ID */
  id: string;
  /** 课程名称 */
  courseName: string;
  /** 考试类型 */
  examType: string;
  /** 考试日期 */
  examDate: string;
  /** 考试时间 */
  examTime: string;
  /** 考试地点 */
  location: string;
  /** 座位号 */
  seatNumber: string;
  /** 学期 */
  semester: string;
}

/**
 * 学分进度接口
 */
export interface ICreditProgress {
  /** 学分类别 */
  category: string;
  /** 要求学分 */
  required: number;
  /** 已获得学分 */
  earned: number;
  /** 剩余学分 */
  remaining: number;
  /** 课程列表 */
  courses: ICreditCourse[];
}

/**
 * 学分课程接口
 */
export interface ICreditCourse {
  /** 课程名称 */
  name: string;
  /** 学分 */
  credit: number;
  /** 成绩等级 */
  grade: string;
  /** 学期 */
  semester: string;
}
