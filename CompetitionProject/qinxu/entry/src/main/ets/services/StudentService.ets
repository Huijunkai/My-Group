/**
 * 学生数据服务模块
 * 
 * 提供学生数据同步和获取功能，包括：
 * - 学生数据同步
 * - 学期数据获取
 * - 最新学期查询
 * 
 * @module StudentService
 */

import http from '@ohos.net.http';
import { ApiConstants } from '../common/constants/ApiConstants';

/**
 * 同步响应接口
 */
export interface ISyncResponse {
  /** 是否成功 */
  success: boolean;
  /** 消息 */
  message?: string;
  /** 学生信息 */
  student?: Record<string, Object>;
  /** 课表数量 */
  timetableCount?: number;
}

/**
 * 学期数据接口
 */
export interface ISemesterData {
  /** 学期字符串 */
  semester: string;
}

/**
 * 最新学期响应接口
 */
export interface ILatestSemesterResponse {
  /** 是否成功 */
  success: boolean;
  /** 学期数据 */
  data?: ISemesterData;
}

/**
 * 学生数据接口
 */
export interface IStudentData {
  /** 学生信息 */
  info?: Record<string, Object>;
  /** 课程列表 */
  courses?: Object[];
  /** 成绩列表 */
  grades?: Object[];
  /** 考试列表 */
  exams?: Object[];
  /** 计划列表 */
  plans?: Object[];
  /** 进度列表 */
  progress?: Object[];
  /** 学期 */
  semester?: string;
}

/**
 * 学生数据响应接口
 */
export interface IStudentDataResponse {
  /** 是否成功 */
  success: boolean;
  /** 学生数据 */
  data?: IStudentData;
  /** 消息 */
  message?: string;
}

/**
 * 学生数据服务类
 * 
 * 提供学生数据的同步和获取功能
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const studentService = StudentService.getInstance();
 * const result = await studentService.syncStudent('username', 'password');
 * const data = await studentService.getStudentData('studentId');
 * ```
 */
export class StudentService {
  /** 单例实例 */
  private static instance: StudentService;
  /** HTTP请求对象 */
  private httpRequest: http.HttpRequest;

  private constructor() {
    this.httpRequest = http.createHttp();
  }

  static getInstance(): StudentService {
    if (!StudentService.instance) {
      StudentService.instance = new StudentService();
    }
    return StudentService.instance;
  }

  async syncStudent(username: string, password: string, semester?: string): Promise<ISyncResponse> {
    const url = ApiConstants.getFullUrl(ApiConstants.ENDPOINTS.SYNC);
    const body: Record<string, Object> = {
      username: username,
      password: password
    };
    
    if (semester) {
      body['semester'] = semester;
    }

    try {
      const response: http.HttpResponse = await this.httpRequest.request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify(body),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      return JSON.parse(resultString) as ISyncResponse;
    } catch (error) {
      console.error('syncStudent error:', JSON.stringify(error));
      return {
        success: false,
        message: '网络异常，请稍后重试'
      };
    }
  }

  async getStudentData(studentId: string, semester?: string): Promise<IStudentDataResponse> {
    if (!studentId) {
      return { success: false, message: '学号为空' };
    }

    let url = `${ApiConstants.getFullUrl(ApiConstants.ENDPOINTS.STUDENT_DATA)}/${encodeURIComponent(studentId)}`;
    if (semester) {
      url = `${url}?semester=${encodeURIComponent(semester)}`;
    }

    try {
      const response: http.HttpResponse = await this.httpRequest.request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      return JSON.parse(resultString) as IStudentDataResponse;
    } catch (error) {
      console.error('getStudentData error:', JSON.stringify(error));
      return {
        success: false,
        message: '网络异常，请稍后重试'
      };
    }
  }

  async getLatestSemester(): Promise<string | null> {
    const url = ApiConstants.getFullUrl(ApiConstants.ENDPOINTS.LATEST_SEMESTER);

    try {
      const response: http.HttpResponse = await this.httpRequest.request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: ILatestSemesterResponse = JSON.parse(resultString);
      if (json.success && json.data && json.data.semester) {
        return json.data.semester;
      }
      return null;
    } catch (error) {
      console.error('getLatestSemester error:', JSON.stringify(error));
      return null;
    }
  }

  destroy(): void {
    this.httpRequest.destroy();
  }
}
