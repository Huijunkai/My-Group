/**
 * 加密工具模块
 * 
 * 提供数据加密和解密功能，包括：
 * - AES加密/解密
 * - 密钥管理
 * - 安全数据传输
 * 
 * @module CryptoUtils
 */

import cryptoFramework from '@ohos.security.cryptoFramework';
import util from '@ohos.util';
import http from '@ohos.net.http';

/** API基础URL */
const BASE_URL: string = 'http://124.70.92.199:3001';
/** 加密前缀标识 */
const ENCRYPTION_PREFIX: string = 'ENC:';

/**
 * 加密密钥接口
 */
interface EncryptionKey {
  /** 密钥 */
  key: string;
  /** 初始向量 */
  iv: string;
  /** 算法 */
  algorithm: string;
}

/**
 * 密钥响应接口
 */
interface KeyResponse {
  /** 是否成功 */
  success: boolean;
  /** 密钥数据 */
  data?: EncryptionKey;
}

/**
 * 学生信息数据接口
 */
export interface StudentInfoData {
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
 * 课程数据接口
 */
export interface CourseData {
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
  /** 课程类型 */
  courseType?: string;
  /** 原始数据 */
  raw?: string;
}

/**
 * 成绩数据接口
 */
export interface GradeData {
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
 * 考试数据接口
 */
export interface ExamData {
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
 * 计划数据接口
 */
export interface PlanData {
  /** 学期 */
  semester: string;
  /** 课程代码 */
  courseCode: string;
  /** 课程名称 */
  courseName: string;
  /** 教学单位 */
  teachingUnit: string;
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
  /** 是否考试 */
  isExam: string;
}

/**
 * 进度数据接口
 */
export interface ProgressData {
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
 * 电费提醒数据接口
 */
export interface ElectricityReminderData {
  /** 电费账号 */
  electricityAccount: string;
  /** 房间ID */
  roomId: string;
  /** 校区ID */
  campusId: string;
  /** 楼栋ID */
  buildingId: string;
}

/**
 * 加密工具类
 * 
 * 提供AES加密和解密功能
 * 
 * @example
 * ```typescript
 * await CryptoUtils.initialize();
 * const encrypted = await CryptoUtils.encrypt('敏感数据');
 * const decrypted = await CryptoUtils.decrypt(encrypted);
 * ```
 */
export class CryptoUtils {
  /** 加密密钥 */
  private static key: Uint8Array | null = null;
  /** 初始向量 */
  private static iv: Uint8Array | null = null;
  /** 加密器 */
  private static cipher: cryptoFramework.Cipher | null = null;
  /** 对称密钥 */
  private static symKey: cryptoFramework.SymKey | null = null;
  /** 是否已初始化 */
  private static initialized: boolean = false;

  /**
   * 初始化加密工具
   * 
   * @returns 是否初始化成功
   */
  static async initialize(): Promise<boolean> {
    if (CryptoUtils.initialized) {
      return true;
    }

    try {
      const keyData = await CryptoUtils.fetchEncryptionKey();
      if (!keyData) {
        console.error('CryptoUtils: Failed to fetch encryption key');
        return false;
      }

      CryptoUtils.key = CryptoUtils.base64ToUint8Array(keyData.key);
      CryptoUtils.iv = CryptoUtils.base64ToUint8Array(keyData.iv);

      const symKeyGenerator = cryptoFramework.createSymKeyGenerator('AES256');
      const keyBlob: cryptoFramework.DataBlob = {
        data: CryptoUtils.key
      };
      CryptoUtils.symKey = await symKeyGenerator.convertKey(keyBlob);

      CryptoUtils.cipher = cryptoFramework.createCipher('AES256|CBC|PKCS7');

      CryptoUtils.initialized = true;
      console.log('CryptoUtils: Initialized successfully');
      return true;
    } catch (error) {
      console.error('CryptoUtils: Initialization failed:', JSON.stringify(error));
      return false;
    }
  }

  private static async fetchEncryptionKey(): Promise<EncryptionKey | null> {
    const httpRequest = http.createHttp();
    try {
      const response = await httpRequest.request(`${BASE_URL}/api/encryption/key`, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING
      });

      const resultString: string = typeof response.result === 'string'
        ? response.result as string
        : JSON.stringify(response.result);

      const json: KeyResponse = JSON.parse(resultString);
      if (json.success && json.data) {
        return json.data;
      }
      return null;
    } catch (error) {
      console.error('CryptoUtils: Failed to fetch key:', JSON.stringify(error));
      return null;
    } finally {
      httpRequest.destroy();
    }
  }

  private static base64ToUint8Array(base64: string): Uint8Array {
    const base64Helper = new util.Base64Helper();
    return base64Helper.decodeSync(base64);
  }

  static async decrypt(encryptedText: string): Promise<string> {
    if (!encryptedText || !encryptedText.startsWith(ENCRYPTION_PREFIX)) {
      return encryptedText;
    }

    if (!CryptoUtils.initialized) {
      const initResult = await CryptoUtils.initialize();
      if (!initResult) {
        console.error('CryptoUtils: Not initialized');
        return encryptedText;
      }
    }

    try {
      const encryptedData = encryptedText.slice(ENCRYPTION_PREFIX.length);
      const encryptedBytes = CryptoUtils.base64ToUint8Array(encryptedData);

      const ivParamsSpec: cryptoFramework.IvParamsSpec = {
        algName: 'IvParamsSpec',
        iv: {
          data: CryptoUtils.iv!
        }
      };

      await CryptoUtils.cipher!.init(cryptoFramework.CryptoMode.DECRYPT_MODE, CryptoUtils.symKey!, ivParamsSpec);

      const inputBlob: cryptoFramework.DataBlob = {
        data: encryptedBytes
      };

      const outputBlob = await CryptoUtils.cipher!.doFinal(inputBlob);

      const decoder = util.TextDecoder.create('utf-8');
      const decryptedText = decoder.decodeToString(outputBlob.data);

      return decryptedText;
    } catch (error) {
      console.error('CryptoUtils: Decryption failed:', JSON.stringify(error));
      return encryptedText;
    }
  }

  static isEncrypted(text: string): boolean {
    return text !== null && text !== undefined && String(text).startsWith(ENCRYPTION_PREFIX);
  }

  static async decryptStudentInfo(info: StudentInfoData): Promise<StudentInfoData> {
    if (!info) return info;

    const decrypted: StudentInfoData = {
      name: info.name,
      gender: info.gender,
      enrollmentYear: info.enrollmentYear,
      className: info.className,
      major: info.major,
      college: info.college
    };

    if (decrypted.name && CryptoUtils.isEncrypted(decrypted.name)) {
      decrypted.name = await CryptoUtils.decrypt(decrypted.name);
    }
    if (decrypted.gender && CryptoUtils.isEncrypted(decrypted.gender)) {
      decrypted.gender = await CryptoUtils.decrypt(decrypted.gender);
    }
    if (decrypted.enrollmentYear && CryptoUtils.isEncrypted(decrypted.enrollmentYear)) {
      decrypted.enrollmentYear = await CryptoUtils.decrypt(decrypted.enrollmentYear);
    }
    if (decrypted.className && CryptoUtils.isEncrypted(decrypted.className)) {
      decrypted.className = await CryptoUtils.decrypt(decrypted.className);
    }
    if (decrypted.major && CryptoUtils.isEncrypted(decrypted.major)) {
      decrypted.major = await CryptoUtils.decrypt(decrypted.major);
    }
    if (decrypted.college && CryptoUtils.isEncrypted(decrypted.college)) {
      decrypted.college = await CryptoUtils.decrypt(decrypted.college);
    }

    return decrypted;
  }

  static async decryptCourse(course: CourseData): Promise<CourseData> {
    if (!course) return course;

    const decrypted: CourseData = {
      dayOfWeek: course.dayOfWeek,
      week: course.week,
      weeks: course.weeks,
      period: course.period,
      name: course.name,
      location: course.location,
      teacher: course.teacher,
      semester: course.semester,
      courseType: course.courseType,
      raw: course.raw
    };

    if (decrypted.name && CryptoUtils.isEncrypted(decrypted.name)) {
      decrypted.name = await CryptoUtils.decrypt(decrypted.name);
    }
    if (decrypted.teacher && CryptoUtils.isEncrypted(decrypted.teacher)) {
      decrypted.teacher = await CryptoUtils.decrypt(decrypted.teacher);
    }
    if (decrypted.location && CryptoUtils.isEncrypted(decrypted.location)) {
      decrypted.location = await CryptoUtils.decrypt(decrypted.location);
    }
    if (decrypted.weeks && CryptoUtils.isEncrypted(decrypted.weeks)) {
      decrypted.weeks = await CryptoUtils.decrypt(decrypted.weeks);
    }
    if (decrypted.courseType && CryptoUtils.isEncrypted(decrypted.courseType)) {
      decrypted.courseType = await CryptoUtils.decrypt(decrypted.courseType);
    }
    if (decrypted.raw && CryptoUtils.isEncrypted(decrypted.raw)) {
      decrypted.raw = await CryptoUtils.decrypt(decrypted.raw);
    }

    return decrypted;
  }

  static async decryptGrade(grade: GradeData): Promise<GradeData> {
    if (!grade) return grade;

    const decrypted: GradeData = {
      semester: grade.semester,
      courseCode: grade.courseCode,
      courseName: grade.courseName,
      score: grade.score,
      credit: grade.credit,
      gradePoint: grade.gradePoint,
      courseType: grade.courseType,
      examType: grade.examType
    };

    if (decrypted.courseName && CryptoUtils.isEncrypted(decrypted.courseName)) {
      decrypted.courseName = await CryptoUtils.decrypt(decrypted.courseName);
    }
    if (decrypted.score && CryptoUtils.isEncrypted(decrypted.score)) {
      decrypted.score = await CryptoUtils.decrypt(decrypted.score);
    }
    if (decrypted.credit && CryptoUtils.isEncrypted(decrypted.credit)) {
      decrypted.credit = await CryptoUtils.decrypt(decrypted.credit);
    }
    if (decrypted.gradePoint && CryptoUtils.isEncrypted(decrypted.gradePoint)) {
      decrypted.gradePoint = await CryptoUtils.decrypt(decrypted.gradePoint);
    }
    if (decrypted.courseType && CryptoUtils.isEncrypted(decrypted.courseType)) {
      decrypted.courseType = await CryptoUtils.decrypt(decrypted.courseType);
    }
    if (decrypted.examType && CryptoUtils.isEncrypted(decrypted.examType)) {
      decrypted.examType = await CryptoUtils.decrypt(decrypted.examType);
    }

    return decrypted;
  }

  static async decryptExam(exam: ExamData): Promise<ExamData> {
    if (!exam) return exam;

    const decrypted: ExamData = {
      courseName: exam.courseName,
      examTime: exam.examTime,
      location: exam.location,
      seatNumber: exam.seatNumber,
      examType: exam.examType,
      status: exam.status
    };

    if (decrypted.courseName && CryptoUtils.isEncrypted(decrypted.courseName)) {
      decrypted.courseName = await CryptoUtils.decrypt(decrypted.courseName);
    }
    if (decrypted.location && CryptoUtils.isEncrypted(decrypted.location)) {
      decrypted.location = await CryptoUtils.decrypt(decrypted.location);
    }
    if (decrypted.seatNumber && CryptoUtils.isEncrypted(decrypted.seatNumber)) {
      decrypted.seatNumber = await CryptoUtils.decrypt(decrypted.seatNumber);
    }
    if (decrypted.examType && CryptoUtils.isEncrypted(decrypted.examType)) {
      decrypted.examType = await CryptoUtils.decrypt(decrypted.examType);
    }
    if (decrypted.status && CryptoUtils.isEncrypted(decrypted.status)) {
      decrypted.status = await CryptoUtils.decrypt(decrypted.status);
    }

    return decrypted;
  }

  static async decryptPlan(plan: PlanData): Promise<PlanData> {
    if (!plan) return plan;

    const decrypted: PlanData = {
      semester: plan.semester,
      courseCode: plan.courseCode,
      courseName: plan.courseName,
      teachingUnit: plan.teachingUnit,
      credit: plan.credit,
      totalHours: plan.totalHours,
      courseType: plan.courseType,
      courseAttribute: plan.courseAttribute,
      examType: plan.examType,
      isExam: plan.isExam
    };

    if (decrypted.courseName && CryptoUtils.isEncrypted(decrypted.courseName)) {
      decrypted.courseName = await CryptoUtils.decrypt(decrypted.courseName);
    }
    if (decrypted.teachingUnit && CryptoUtils.isEncrypted(decrypted.teachingUnit)) {
      decrypted.teachingUnit = await CryptoUtils.decrypt(decrypted.teachingUnit);
    }
    if (decrypted.credit && CryptoUtils.isEncrypted(decrypted.credit)) {
      decrypted.credit = await CryptoUtils.decrypt(decrypted.credit);
    }
    if (decrypted.totalHours && CryptoUtils.isEncrypted(decrypted.totalHours)) {
      decrypted.totalHours = await CryptoUtils.decrypt(decrypted.totalHours);
    }
    if (decrypted.courseType && CryptoUtils.isEncrypted(decrypted.courseType)) {
      decrypted.courseType = await CryptoUtils.decrypt(decrypted.courseType);
    }
    if (decrypted.courseAttribute && CryptoUtils.isEncrypted(decrypted.courseAttribute)) {
      decrypted.courseAttribute = await CryptoUtils.decrypt(decrypted.courseAttribute);
    }
    if (decrypted.examType && CryptoUtils.isEncrypted(decrypted.examType)) {
      decrypted.examType = await CryptoUtils.decrypt(decrypted.examType);
    }
    if (decrypted.isExam && CryptoUtils.isEncrypted(decrypted.isExam)) {
      decrypted.isExam = await CryptoUtils.decrypt(decrypted.isExam);
    }

    return decrypted;
  }

  static async decryptProgress(progress: ProgressData): Promise<ProgressData> {
    if (!progress) return progress;

    const decrypted: ProgressData = {
      category: progress.category,
      requiredCredits: progress.requiredCredits,
      completedCredits: progress.completedCredits,
      currentCredits: progress.currentCredits,
      remainingCredits: progress.remainingCredits
    };

    if (decrypted.category && CryptoUtils.isEncrypted(decrypted.category)) {
      decrypted.category = await CryptoUtils.decrypt(decrypted.category);
    }
    if (decrypted.requiredCredits && CryptoUtils.isEncrypted(decrypted.requiredCredits)) {
      decrypted.requiredCredits = await CryptoUtils.decrypt(decrypted.requiredCredits);
    }
    if (decrypted.completedCredits && CryptoUtils.isEncrypted(decrypted.completedCredits)) {
      decrypted.completedCredits = await CryptoUtils.decrypt(decrypted.completedCredits);
    }
    if (decrypted.currentCredits && CryptoUtils.isEncrypted(decrypted.currentCredits)) {
      decrypted.currentCredits = await CryptoUtils.decrypt(decrypted.currentCredits);
    }
    if (decrypted.remainingCredits && CryptoUtils.isEncrypted(decrypted.remainingCredits)) {
      decrypted.remainingCredits = await CryptoUtils.decrypt(decrypted.remainingCredits);
    }

    return decrypted;
  }

  static async decryptElectricityReminder(setting: ElectricityReminderData): Promise<ElectricityReminderData> {
    if (!setting) return setting;

    const decrypted: ElectricityReminderData = {
      electricityAccount: setting.electricityAccount,
      roomId: setting.roomId,
      campusId: setting.campusId,
      buildingId: setting.buildingId
    };

    if (decrypted.electricityAccount && CryptoUtils.isEncrypted(decrypted.electricityAccount)) {
      decrypted.electricityAccount = await CryptoUtils.decrypt(decrypted.electricityAccount);
    }
    if (decrypted.roomId && CryptoUtils.isEncrypted(decrypted.roomId)) {
      decrypted.roomId = await CryptoUtils.decrypt(decrypted.roomId);
    }
    if (decrypted.campusId && CryptoUtils.isEncrypted(decrypted.campusId)) {
      decrypted.campusId = await CryptoUtils.decrypt(decrypted.campusId);
    }
    if (decrypted.buildingId && CryptoUtils.isEncrypted(decrypted.buildingId)) {
      decrypted.buildingId = await CryptoUtils.decrypt(decrypted.buildingId);
    }

    return decrypted;
  }

  static async decryptCoursesArray(courses: CourseData[]): Promise<CourseData[]> {
    if (!courses || !Array.isArray(courses)) return courses;

    const decryptedCourses: CourseData[] = [];
    for (let i = 0; i < courses.length; i++) {
      const decrypted = await CryptoUtils.decryptCourse(courses[i]);
      decryptedCourses.push(decrypted);
    }

    return decryptedCourses;
  }

  static async decryptGradesArray(grades: GradeData[]): Promise<GradeData[]> {
    if (!grades || !Array.isArray(grades)) return grades;

    const decryptedGrades: GradeData[] = [];
    for (let i = 0; i < grades.length; i++) {
      const decrypted = await CryptoUtils.decryptGrade(grades[i]);
      decryptedGrades.push(decrypted);
    }

    return decryptedGrades;
  }

  static async decryptExamsArray(exams: ExamData[]): Promise<ExamData[]> {
    if (!exams || !Array.isArray(exams)) return exams;

    const decryptedExams: ExamData[] = [];
    for (let i = 0; i < exams.length; i++) {
      const decrypted = await CryptoUtils.decryptExam(exams[i]);
      decryptedExams.push(decrypted);
    }

    return decryptedExams;
  }

  static async decryptPlansArray(plans: PlanData[]): Promise<PlanData[]> {
    if (!plans || !Array.isArray(plans)) return plans;

    const decryptedPlans: PlanData[] = [];
    for (let i = 0; i < plans.length; i++) {
      const decrypted = await CryptoUtils.decryptPlan(plans[i]);
      decryptedPlans.push(decrypted);
    }

    return decryptedPlans;
  }

  static async decryptProgressArray(progressArray: ProgressData[]): Promise<ProgressData[]> {
    if (!progressArray || !Array.isArray(progressArray)) return progressArray;

    const decryptedProgress: ProgressData[] = [];
    for (let i = 0; i < progressArray.length; i++) {
      const decrypted = await CryptoUtils.decryptProgress(progressArray[i]);
      decryptedProgress.push(decrypted);
    }

    return decryptedProgress;
  }
}
