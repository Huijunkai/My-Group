/**
 * API客户端模块
 * 
 * 提供与后端服务器通信的功能，包括：
 * - 学生数据同步
 * - 公告获取
 * - 空教室查询
 * 
 * @module ApiClient
 */

import http from '@ohos.net.http';
import { DataStore, StudentInfo, Grade, Exam, Plan, Progress, ScheduleData } from './DataStore';
import { CryptoUtils, StudentInfoData, CourseData, GradeData, ExamData, PlanData, ProgressData } from './CryptoUtils';

/** API基础URL */
const BASE_URL: string = 'http://124.70.92.199:3000';

/**
 * 同步数据接口
 */
interface SyncData {
  /** 学生信息 */
  info?: StudentInfoData;
  /** 课程列表 */
  courses?: CourseData[];
  /** 成绩列表 */
  grades?: SemesterGrades[];
  /** 考试列表 */
  exams?: ExamData[];
  /** 计划列表 */
  plans?: SemesterPlans[];
  /** 进度列表 */
  progress?: ProgressData[];
  /** 学期 */
  semester?: string;
  /** 最后更新时间 */
  lastUpdated?: number;
}

/**
 * 学期成绩接口
 */
interface SemesterGrades {
  /** 学期 */
  semester: string;
  /** 成绩列表 */
  grades: GradeData[];
}

/**
 * 学期计划接口
 */
interface SemesterPlans {
  /** 学期 */
  semester: string;
  /** 计划列表 */
  plans: PlanData[];
}

/**
 * 同步响应接口
 */
interface SyncResponse {
  /** 是否成功 */
  success: boolean;
  /** 消息 */
  message?: string;
  /** 数据 */
  data?: SyncData;
}

/**
 * 最新学期数据接口
 */
interface LatestSemesterData {
  /** 学期 */
  semester: string;
}

/**
 * 最新学期响应接口
 */
interface LatestSemesterResponse {
  /** 是否成功 */
  success: boolean;
  /** 数据 */
  data?: LatestSemesterData;
}

/**
 * 变更检测结果接口
 */
interface HasChangesResult {
  /** 信息变更 */
  info: boolean;
  /** 课程变更 */
  courses: boolean;
  /** 成绩变更 */
  grades: boolean;
  /** 考试变更 */
  exams: boolean;
  /** 计划变更 */
  plans: boolean;
  /** 进度变更 */
  progress: boolean;
}

/**
 * 计数结果接口
 */
interface CountsResult {
  /** 学生信息 */
  info?: StudentInfo;
  /** 课程数 */
  courses?: number;
  /** 成绩数 */
  grades?: number;
  /** 考试数 */
  exams?: number;
  /** 计划数 */
  plans?: number;
  /** 进度数 */
  progress?: number;
}

/**
 * 保存数据结果接口
 */
interface SaveDataResult {
  /** 变更检测 */
  hasChanges: HasChangesResult;
  /** 计数结果 */
  counts: CountsResult;
}

/**
 * 同步结果接口
 */
export interface SyncResult {
  /** 是否成功 */
  success: boolean;
  /** 消息 */
  message?: string;
  /** 变更检测 */
  hasChanges?: HasChangesResult;
  /** 数据 */
  data?: CountsResult;
}

/**
 * 公告接口
 */
export interface Announcement {
  /** 公告ID */
  id: number;
  /** 标题 */
  title: string;
  /** URL */
  url: string;
  /** 日期 */
  date: string;
}

/**
 * 公告附件接口
 */
export interface AnnouncementAttachment {
  /** 附件名称 */
  name: string;
  /** 附件URL */
  url: string;
}

/**
 * 公告详情接口
 */
export interface AnnouncementDetail {
  /** 标题 */
  title: string;
  /** 内容 */
  content: string;
  /** 日期 */
  date: string;
  /** 附件列表 */
  attachments: AnnouncementAttachment[];
  /** URL */
  url: string;
}

/**
 * 空教室查询参数接口
 */
export interface EmptyRoomQuery {
  semester: string;
  campus?: string;
  building?: string;
  weekStart: number;
  weekEnd: number;
  periodStart?: string;
  periodEnd?: string;
}

export interface EmptyRoom {
  roomName: string;
  building: string;
  campus: string;
  capacity: string;
  type: string;
  emptySlots?: EmptySlot[];
}

export interface EmptySlot {
  day: number;
  periods: number[];
}

export interface RoomScheduleQuery {
  roomName: string;
  semester: string;
  campus?: string;
  building?: string;
  weekStart: number;
  weekEnd: number;
  periodStart?: string;
  periodEnd?: string;
}

export interface DaySchedule {
  day: number;
  periods: string[];
}

export interface RoomSchedule {
  roomName: string;
  schedule: DaySchedule[];
}

export interface WaterScanResult {
  location: string;
  userid: string;
  saler: string;
  openid?: string;
  deviceid?: string;
  balance?: string;
  cardNo?: string;
  ownerName?: string;
}

interface WaterScanResponse {
  success: boolean;
  message?: string;
  data?: WaterScanResult;
}

interface AnnouncementResponse {
  success: boolean;
  data?: Announcement[];
  message?: string;
}

interface PaginatedAnnouncementResponse {
  success: boolean;
  data?: Announcement[];
  total: number;
  message?: string;
}

interface AnnouncementsResult {
  announcements: Announcement[];
  total: number;
}

interface AnnouncementDetailResponse {
  success: boolean;
  data?: AnnouncementDetail;
  message?: string;
}

interface EmptyRoomResponse {
  success: boolean;
  data?: EmptyRoom[];
  message?: string;
}

interface RoomScheduleResponse {
  success: boolean;
  data?: RoomSchedule;
  message?: string;
}

interface CampusResponse {
  success: boolean;
  data?: CampusItem[];
  message?: string;
}

interface BuildingResponse {
  success: boolean;
  data?: BuildingItem[];
  message?: string;
}

export interface CampusItem {
  code: string;
  name: string;
}

export interface BuildingItem {
  code: string;
  name: string;
}

interface RequestBody {
  username: string;
  password: string;
  semester?: string;
}

export class ApiClient {
  private static httpRequest: http.HttpRequest | null = null;
  private static dataStore: DataStore = DataStore.getInstance();

  private static getHttpRequest(): http.HttpRequest {
    if (!ApiClient.httpRequest) {
      ApiClient.httpRequest = http.createHttp();
    }
    return ApiClient.httpRequest;
  }

  private static safeJsonParse<T>(jsonString: string, defaultValue: T): T {
    try {
      if (!jsonString || typeof jsonString !== 'string') {
        console.warn('ApiClient: Invalid input for JSON parse');
        return defaultValue;
      }
      return JSON.parse(jsonString) as T;
    } catch (error) {
      console.error('ApiClient: JSON parse error:', error, 'Input:', jsonString.substring(0, 100));
      return defaultValue;
    }
  }

  static async destroy(): Promise<void> {
    if (ApiClient.httpRequest) {
      try {
        await ApiClient.httpRequest.destroy();
        ApiClient.httpRequest = null;
        console.log('ApiClient: HttpRequest destroyed successfully');
      } catch (error) {
        console.error('ApiClient: Failed to destroy HttpRequest:', error);
      }
    }
  }

  static async syncStudent(username: string, password: string, semester?: string): Promise<SyncResult> {
    const url = `${BASE_URL}/api/sync`;
    const body: RequestBody = {
      username: username,
      password: password
    };
    if (semester) {
      body.semester = semester;
    }

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
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

      const json = ApiClient.safeJsonParse<SyncResponse>(resultString, { success: false, message: '数据解析失败' });

      if (!json.success || !json.data) {
        const result: SyncResult = {
          success: false,
          message: json.message ? json.message : '同步失败'
        };
        return result;
      }

      const changes = await ApiClient.saveDataToLocal(json.data, username);

      const result: SyncResult = {
        success: true,
        message: json.message,
        hasChanges: changes.hasChanges,
        data: changes.counts
      };
      return result;
    } catch (error) {
      console.error('syncStudent error:', JSON.stringify(error));
      const result: SyncResult = {
        success: false,
        message: '网络异常，请稍后重试'
      };
      return result;
    }
  }

  private static async saveDataToLocal(data: SyncData, studentId: string): Promise<SaveDataResult> {
    const hasChanges: HasChangesResult = {
      info: false,
      courses: false,
      grades: false,
      exams: false,
      plans: false,
      progress: false
    };
    const counts: CountsResult = {};

    await CryptoUtils.initialize();

    if (data.info) {
      const decryptedInfo = await CryptoUtils.decryptStudentInfo(data.info);
      const newInfo = ApiClient.parseStudentInfo(decryptedInfo, studentId);
      const oldInfo = await ApiClient.dataStore.loadStudentInfo();
      if (!oldInfo || ApiClient.hasInfoChanged(oldInfo, newInfo)) {
        await ApiClient.dataStore.saveStudentInfo(newInfo);
        hasChanges.info = true;
      }
      counts.info = newInfo;
    }

    if (data.courses && Array.isArray(data.courses)) {
      const decryptedCourses = await CryptoUtils.decryptCoursesArray(data.courses);
      const newCourses: Object[] = [];
      for (let i = 0; i < decryptedCourses.length; i++) {
        newCourses.push(decryptedCourses[i]);
      }
      const oldScheduleData = await ApiClient.dataStore.loadScheduleData();
      const oldCourses: Object[] = oldScheduleData ? oldScheduleData.courses : [];
      if (ApiClient.hasCoursesChanged(oldCourses, newCourses)) {
        const scheduleData: ScheduleData = {
          courses: newCourses,
          semester: data.semester ? data.semester : '',
          lastUpdated: data.lastUpdated ? data.lastUpdated : Date.now()
        };
        await ApiClient.dataStore.saveScheduleData(scheduleData);
        hasChanges.courses = true;
      }
      counts.courses = newCourses.length;
    }

    if (data.grades && Array.isArray(data.grades)) {
      const allGrades: Grade[] = [];
      for (let i = 0; i < data.grades.length; i++) {
        const semesterGrades = data.grades[i];
        const decryptedArray = await CryptoUtils.decryptGradesArray(semesterGrades.grades);
        for (let j = 0; j < decryptedArray.length; j++) {
          const g = decryptedArray[j];
          const grade: Grade = {
            semester: semesterGrades.semester,
            courseCode: g.courseCode,
            courseName: g.courseName,
            score: g.score,
            credit: g.credit,
            gradePoint: g.gradePoint,
            courseType: g.courseType,
            examType: g.examType
          };
          allGrades.push(grade);
        }
      }
      const oldGrades = await ApiClient.dataStore.loadGrades();
      if (ApiClient.hasGradesChanged(oldGrades, allGrades)) {
        await ApiClient.dataStore.saveGrades(allGrades);
        hasChanges.grades = true;
      }
      counts.grades = allGrades.length;
    }

    if (data.exams && Array.isArray(data.exams)) {
      const decryptedExams = await CryptoUtils.decryptExamsArray(data.exams);
      const newExams: Exam[] = [];
      for (let i = 0; i < decryptedExams.length; i++) {
        const e = decryptedExams[i];
        const exam: Exam = {
          courseName: e.courseName,
          examTime: e.examTime,
          location: e.location,
          seatNumber: e.seatNumber,
          examType: e.examType,
          status: e.status
        };
        newExams.push(exam);
      }
      const oldExams = await ApiClient.dataStore.loadExams();
      if (ApiClient.hasExamsChanged(oldExams, newExams)) {
        await ApiClient.dataStore.saveExams(newExams);
        hasChanges.exams = true;
      }
      counts.exams = newExams.length;
    }

    if (data.plans && Array.isArray(data.plans)) {
      const allPlans: Plan[] = [];
      for (let i = 0; i < data.plans.length; i++) {
        const semesterPlans = data.plans[i];
        const decryptedArray = await CryptoUtils.decryptPlansArray(semesterPlans.plans);
        for (let j = 0; j < decryptedArray.length; j++) {
          const p = decryptedArray[j];
          const plan: Plan = {
            semester: semesterPlans.semester,
            courseCode: p.courseCode,
            courseName: p.courseName,
            credit: p.credit,
            totalHours: p.totalHours,
            courseType: p.courseType,
            courseAttribute: p.courseAttribute,
            examType: p.examType
          };
          allPlans.push(plan);
        }
      }
      const oldPlans = await ApiClient.dataStore.loadPlans();
      if (ApiClient.hasPlansChanged(oldPlans, allPlans)) {
        await ApiClient.dataStore.savePlans(allPlans);
        hasChanges.plans = true;
      }
      counts.plans = allPlans.length;
    }

    if (data.progress && Array.isArray(data.progress)) {
      const decryptedProgress = await CryptoUtils.decryptProgressArray(data.progress);
      const newProgress: Progress[] = [];
      for (let i = 0; i < decryptedProgress.length; i++) {
        const p = decryptedProgress[i];
        const progress: Progress = {
          category: p.category ? p.category : '',
          requiredCredits: p.requiredCredits ? p.requiredCredits : '',
          completedCredits: p.completedCredits ? p.completedCredits : '',
          currentCredits: p.currentCredits ? p.currentCredits : '',
          remainingCredits: p.remainingCredits ? p.remainingCredits : ''
        };
        newProgress.push(progress);
      }
      const oldProgress = await ApiClient.dataStore.loadProgress();
      if (ApiClient.hasProgressChanged(oldProgress, newProgress)) {
        await ApiClient.dataStore.saveProgress(newProgress);
        hasChanges.progress = true;
      }
      counts.progress = newProgress.length;
    }

    const result: SaveDataResult = { hasChanges: hasChanges, counts: counts };
    return result;
  }

  private static parseStudentInfo(info: StudentInfoData, studentId: string): StudentInfo {
    const result: StudentInfo = {
      studentId: studentId,
      name: info.name,
      gender: info.gender,
      enrollmentYear: info.enrollmentYear,
      className: info.className,
      major: info.major,
      college: info.college
    };
    return result;
  }

  private static hasInfoChanged(oldInfo: StudentInfo, newInfo: StudentInfo): boolean {
    return oldInfo.name !== newInfo.name ||
      oldInfo.gender !== newInfo.gender ||
      oldInfo.className !== newInfo.className ||
      oldInfo.major !== newInfo.major ||
      oldInfo.college !== newInfo.college;
  }

  private static hasCoursesChanged(oldCourses: Object[], newCourses: Object[]): boolean {
    if (oldCourses.length !== newCourses.length) return true;
    const oldJson = JSON.stringify(oldCourses);
    const newJson = JSON.stringify(newCourses);
    return oldJson !== newJson;
  }

  private static hasGradesChanged(oldGrades: Grade[], newGrades: Grade[]): boolean {
    if (oldGrades.length !== newGrades.length) return true;
    const oldMap = new Map<string, Grade>();
    for (let i = 0; i < oldGrades.length; i++) {
      const g = oldGrades[i];
      oldMap.set(`${g.semester}_${g.courseCode}`, g);
    }
    for (let i = 0; i < newGrades.length; i++) {
      const ng = newGrades[i];
      const key = `${ng.semester}_${ng.courseCode}`;
      const og = oldMap.get(key);
      if (!og || og.score !== ng.score) return true;
    }
    return false;
  }

  private static hasExamsChanged(oldExams: Exam[], newExams: Exam[]): boolean {
    if (oldExams.length !== newExams.length) return true;
    const oldJson = JSON.stringify(oldExams);
    const newJson = JSON.stringify(newExams);
    return oldJson !== newJson;
  }

  private static hasPlansChanged(oldPlans: Plan[], newPlans: Plan[]): boolean {
    if (oldPlans.length !== newPlans.length) return true;
    const oldJson = JSON.stringify(oldPlans);
    const newJson = JSON.stringify(newPlans);
    return oldJson !== newJson;
  }

  private static hasProgressChanged(oldProgress: Progress[], newProgress: Progress[]): boolean {
    if (oldProgress.length !== newProgress.length) return true;
    const oldJson = JSON.stringify(oldProgress);
    const newJson = JSON.stringify(newProgress);
    return oldJson !== newJson;
  }

  static async getLatestSemester(): Promise<string | null> {
    const url = `${BASE_URL}/api/semester/latest`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: LatestSemesterResponse = ApiClient.safeJsonParse<LatestSemesterResponse>(resultString, { success: false });
      if (json.success && json.data && json.data.semester) {
        return json.data.semester;
      }
      return null;
    } catch (error) {
      console.error('getLatestSemester error:', JSON.stringify(error));
      return null;
    }
  }

  static async getAnnouncements(limit: number = 5, offset: number = 0): Promise<AnnouncementsResult> {
    const url = `${BASE_URL}/api/announcements?limit=${limit}&offset=${offset}`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: PaginatedAnnouncementResponse = ApiClient.safeJsonParse<PaginatedAnnouncementResponse>(resultString, { success: false, data: [], total: 0 });
      if (json.success && json.data) {
        return {
          announcements: json.data,
          total: json.total || 0
        };
      }
      return {
        announcements: [],
        total: 0
      };
    } catch (error) {
      console.error('getAnnouncements error:', JSON.stringify(error));
      return {
        announcements: [],
        total: 0
      };
    }
  }

  static async getAnnouncementDetail(url: string): Promise<AnnouncementDetail | null> {
    const apiUrl = `${BASE_URL}/api/announcements/detail?url=${encodeURIComponent(url)}`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(apiUrl, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: AnnouncementDetailResponse = ApiClient.safeJsonParse<AnnouncementDetailResponse>(resultString, { success: false });
      if (json.success && json.data) {
        return json.data;
      }
      return null;
    } catch (error) {
      console.error('getAnnouncementDetail error:', JSON.stringify(error));
      return null;
    }
  }

  static async getCampuses(): Promise<CampusItem[]> {
    const url = `${BASE_URL}/api/emptyroom/campuses`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: CampusResponse = ApiClient.safeJsonParse<CampusResponse>(resultString, { success: false });
      if (json.success && json.data) {
        return json.data;
      }
      return [];
    } catch (error) {
      console.error('getCampuses error:', JSON.stringify(error));
      return [];
    }
  }

  static async getBuildings(campusCode: string): Promise<BuildingItem[]> {
    const url = `${BASE_URL}/api/emptyroom/buildings?campus=${campusCode}`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: BuildingResponse = ApiClient.safeJsonParse<BuildingResponse>(resultString, { success: false });
      if (json.success && json.data) {
        return json.data;
      }
      return [];
    } catch (error) {
      console.error('getBuildings error:', JSON.stringify(error));
      return [];
    }
  }

  static async queryEmptyRooms(query: EmptyRoomQuery): Promise<EmptyRoom[]> {
    const url = `${BASE_URL}/api/emptyroom/query`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify(query),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: EmptyRoomResponse = ApiClient.safeJsonParse<EmptyRoomResponse>(resultString, { success: false });
      if (json.success && json.data) {
        return json.data;
      }
      return [];
    } catch (error) {
      console.error('queryEmptyRooms error:', JSON.stringify(error));
      return [];
    }
  }

  static async queryRoomSchedule(query: RoomScheduleQuery): Promise<RoomSchedule | null> {
    const url = `${BASE_URL}/api/emptyroom/schedule`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify(query),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: RoomScheduleResponse = ApiClient.safeJsonParse<RoomScheduleResponse>(resultString, { success: false });
      if (json.success && json.data) {
        return json.data;
      }
      return null;
    } catch (error) {
      console.error('queryRoomSchedule error:', JSON.stringify(error));
      return null;
    }
  }

  static async scanWaterQrcode(scanUrl: string): Promise<WaterScanResponse> {
    const url = `${BASE_URL}/api/water/scan`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify({ scanUrl: scanUrl }),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: WaterScanResponse = ApiClient.safeJsonParse<WaterScanResponse>(resultString, { success: false, message: '网络异常' });
      return json;
    } catch (error) {
      console.error('scanWaterQrcode error:', JSON.stringify(error));
      const result: WaterScanResponse = {
        success: false,
        message: '网络异常，请稍后重试'
      };
      return result;
    }
  }

  static async registerPushToken(studentId: string, pushToken: string): Promise<SyncResult> {
    const url = `${BASE_URL}/api/push/register`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify({ studentId: studentId, pushToken: pushToken }),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: SyncResponse = ApiClient.safeJsonParse<SyncResponse>(resultString, { success: false });
      const result: SyncResult = {
        success: json.success,
        message: json.message
      };
      return result;
    } catch (error) {
      console.error('registerPushToken error:', JSON.stringify(error));
      const result: SyncResult = {
        success: false,
        message: '网络异常，请稍后重试'
      };
      return result;
    }
  }

  static async unregisterPushToken(studentId: string): Promise<SyncResult> {
    const url = `${BASE_URL}/api/push/unregister`;

    try {
      const response: http.HttpResponse = await ApiClient.getHttpRequest().request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify({ studentId: studentId }),
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: SyncResponse = ApiClient.safeJsonParse<SyncResponse>(resultString, { success: false });
      const result: SyncResult = {
        success: json.success,
        message: json.message
      };
      return result;
    } catch (error) {
      console.error('unregisterPushToken error:', JSON.stringify(error));
      const result: SyncResult = {
        success: false,
        message: '网络异常，请稍后重试'
      };
      return result;
    }
  }
}
