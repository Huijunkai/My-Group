/**
 * 学期工具模块
 * 
 * 提供学期信息解析和管理功能，包括：
 * - 学期字符串解析
 * - 学期显示名称格式化
 * - 学期排序
 * - 学生学期计算
 * 
 * @module SemesterUtils
 */

/**
 * 学期信息接口
 */
export interface SemesterInfo {
  /** 原始字符串 */
  raw: string;
  /** 显示名称 */
  displayName: string;
  /** 年份 */
  year: number;
  /** 学期号 */
  semester: number;
  /** 开始年份 */
  startYear: number;
  /** 结束年份 */
  endYear: number;
  /** 排序键 */
  sortKey: number;
}

/**
 * 学生学期信息接口
 */
export interface StudentSemesterInfo {
  /** 学期信息 */
  semester: SemesterInfo;
  /** 学期序号 */
  semesterNumber: number;
}

/**
 * 带学期的成绩接口
 */
interface GradeWithSemester {
  /** 学期字符串 */
  semester?: string;
}

/** 学期匹配正则表达式列表 */
const SEMESTER_PATTERNS: RegExp[] = [
  /^(\d{4})-(\d{4})-(\d)$/,
  /^(\d{4})-(\d{4})学年第([一二三四])学期$/,
  /^(\d{4})-(\d{4})学年第([1234])学期$/,
  /^(\d{4})年([春夏秋冬])季学期$/
];

const SEMESTER_NAMES: Record<string, string> = {
  '1': '第一学期',
  '2': '第二学期',
  '3': '第三学期',
  '4': '第四学期',
  '一': '第一学期',
  '二': '第二学期',
  '三': '第三学期',
  '四': '第四学期',
  '春': '春季学期',
  '夏': '夏季学期',
  '秋': '秋季学期',
  '冬': '冬季学期'
};

const GRADE_NAMES: Record<number, string> = {
  1: '大一上学期',
  2: '大一下学期',
  3: '大二上学期',
  4: '大二下学期',
  5: '大三上学期',
  6: '大三下学期',
  7: '大四上学期',
  8: '大四下学期'
};

function convertChineseToNumber(chinese: string): number {
  const map: Record<string, number> = {
    '一': 1, '二': 2, '三': 3, '四': 4
  };
  return map[chinese] || 1;
}

function convertSeasonToNumber(season: string): number {
  const map: Record<string, number> = {
    '春': 1, '夏': 2, '秋': 1, '冬': 2
  };
  return map[season] || 1;
}

function formatDisplayName(startYear: number, endYear: number, semesterNum: number): string {
  const semesterName = SEMESTER_NAMES[String(semesterNum)] || `第${semesterNum}学期`;
  return `${startYear}-${endYear}学年${semesterName}`;
}

function parseFallback(semesterStr: string): SemesterInfo | null {
  const yearMatch = semesterStr.match(/(\d{4})/);
  if (yearMatch) {
    const year = parseInt(yearMatch[1]);
    const semesterNum = semesterStr.includes('二') || semesterStr.includes('2') ? 2 : 1;
    const sortKey = year * 10 + semesterNum;
    const displayName = formatDisplayName(year, year + 1, semesterNum);

    const result: SemesterInfo = {
      raw: semesterStr,
      displayName: displayName,
      year: year,
      semester: semesterNum,
      startYear: year,
      endYear: year + 1,
      sortKey: sortKey
    };
    return result;
  }

  return null;
}

export function parseSemester(semesterStr: string): SemesterInfo | null {
  if (!semesterStr || semesterStr === '未知学期') {
    return null;
  }

  let match: RegExpExecArray | null = null;
  let patternIndex = -1;

  for (let i = 0; i < SEMESTER_PATTERNS.length; i++) {
    match = SEMESTER_PATTERNS[i].exec(semesterStr);
    if (match) {
      patternIndex = i;
      break;
    }
  }

  if (!match) {
    return parseFallback(semesterStr);
  }

  let startYear: number;
  let endYear: number;
  let semesterNum: number;

  if (patternIndex === 0) {
    startYear = parseInt(match[1]);
    endYear = parseInt(match[2]);
    semesterNum = parseInt(match[3]);
  } else if (patternIndex === 1) {
    startYear = parseInt(match[1]);
    endYear = parseInt(match[2]);
    semesterNum = convertChineseToNumber(match[3]);
  } else if (patternIndex === 2) {
    startYear = parseInt(match[1]);
    endYear = parseInt(match[2]);
    semesterNum = parseInt(match[3]);
  } else {
    startYear = parseInt(match[1]);
    endYear = startYear;
    semesterNum = convertSeasonToNumber(match[2]);
  }

  const sortKey = startYear * 10 + semesterNum;
  const displayName = formatDisplayName(startYear, endYear, semesterNum);

  const result: SemesterInfo = {
    raw: semesterStr,
    displayName: displayName,
    year: startYear,
    semester: semesterNum,
    startYear: startYear,
    endYear: endYear,
    sortKey: sortKey
  };
  return result;
}

export function formatSemester(semesterStr: string): string {
  const info = parseSemester(semesterStr);
  return info ? info.displayName : semesterStr;
}

export function sortSemesters(semesterStrs: string[], ascending: boolean = false): string[] {
  return semesterStrs.sort((a: string, b: string): number => {
    const infoA = parseSemester(a);
    const infoB = parseSemester(b);

    if (!infoA && !infoB) return a.localeCompare(b);
    if (!infoA) return 1;
    if (!infoB) return -1;

    return ascending ? infoA.sortKey - infoB.sortKey : infoB.sortKey - infoA.sortKey;
  });
}

export function getCurrentSemester(): SemesterInfo {
  const now = new Date();
  const currentYear = now.getFullYear();
  const month = now.getMonth() + 1;

  let startYear: number;
  let semesterNum: number;

  if (month >= 2 && month <= 8) {
    startYear = currentYear - 1;
    semesterNum = 2;
  } else {
    startYear = currentYear;
    semesterNum = 1;
  }

  const endYear = startYear + 1;
  const sortKey = startYear * 10 + semesterNum;
  const displayName = formatDisplayName(startYear, endYear, semesterNum);

  const result: SemesterInfo = {
    raw: `${startYear}-${endYear}-${semesterNum}`,
    displayName: displayName,
    year: startYear,
    semester: semesterNum,
    startYear: startYear,
    endYear: endYear,
    sortKey: sortKey
  };
  return result;
}

export function getStudentCurrentSemester(enrollmentYear: string): StudentSemesterInfo | null {
  if (!enrollmentYear) {
    return null;
  }

  const yearMatch = enrollmentYear.match(/(\d{4})/);
  if (!yearMatch) {
    return null;
  }

  const enrollment = parseInt(yearMatch[1]);
  if (isNaN(enrollment) || enrollment < 2000 || enrollment > 2100) {
    return null;
  }

  const current = getCurrentSemester();
  const yearsSinceEnrollment = current.startYear - enrollment;
  
  if (yearsSinceEnrollment < 0) {
    const result: StudentSemesterInfo = {
      semester: current,
      semesterNumber: 1
    };
    return result;
  }
  
  if (yearsSinceEnrollment > 10) {
    const result: StudentSemesterInfo = {
      semester: current,
      semesterNumber: 8
    };
    return result;
  }
  
  const semesterNumber = yearsSinceEnrollment * 2 + current.semester;

  const result: StudentSemesterInfo = {
    semester: current,
    semesterNumber: semesterNumber
  };
  return result;
}

export function getSemesterDescription(enrollmentYear?: string): string {
  const current = getCurrentSemester();

  if (enrollmentYear) {
    const studentInfo = getStudentCurrentSemester(enrollmentYear);
    if (studentInfo) {
      const gradeName = getGradeName(studentInfo.semesterNumber);
      return `当前是${current.displayName}，你正处于${gradeName}（第${studentInfo.semesterNumber}学期）`;
    }
  }

  return `当前是${current.displayName}`;
}

export function getGradeName(semesterNumber: number): string {
  return GRADE_NAMES[semesterNumber] || `第${semesterNumber}学期`;
}

export function getLatestSemesterFromGrades(grades: GradeWithSemester[]): SemesterInfo | null {
  if (!grades || grades.length === 0) {
    return null;
  }

  const semesters = new Set<string>();
  for (let i = 0; i < grades.length; i++) {
    if (grades[i].semester) {
      semesters.add(grades[i].semester as string);
    }
  }

  if (semesters.size === 0) {
    return null;
  }

  const sorted = sortSemesters(Array.from(semesters));
  return parseSemester(sorted[0]);
}

export function inferEnrollmentYearFromGrades(grades: GradeWithSemester[]): string | null {
  const earliest = getEarliestSemesterFromGrades(grades);
  if (!earliest) {
    return null;
  }

  if (earliest.semester === 1) {
    return String(earliest.startYear);
  } else {
    return String(earliest.startYear - 1);
  }
}

export function getEarliestSemesterFromGrades(grades: GradeWithSemester[]): SemesterInfo | null {
  if (!grades || grades.length === 0) {
    return null;
  }

  const semesters = new Set<string>();
  for (let i = 0; i < grades.length; i++) {
    if (grades[i].semester) {
      semesters.add(grades[i].semester as string);
    }
  }

  if (semesters.size === 0) {
    return null;
  }

  const sorted = sortSemesters(Array.from(semesters), true);
  return parseSemester(sorted[0]);
}

export function getSemesterCount(grades: GradeWithSemester[]): number {
  const semesters = new Set<string>();
  for (let i = 0; i < grades.length; i++) {
    if (grades[i].semester) {
      semesters.add(grades[i].semester as string);
    }
  }
  return semesters.size;
}

export class SemesterUtils {
  static parse(semesterStr: string): SemesterInfo | null {
    return parseSemester(semesterStr);
  }

  static format(semesterStr: string): string {
    return formatSemester(semesterStr);
  }

  static sort(semesterStrs: string[], ascending: boolean = false): string[] {
    return sortSemesters(semesterStrs, ascending);
  }

  static getCurrentSemester(): SemesterInfo {
    return getCurrentSemester();
  }

  static getStudentCurrentSemester(enrollmentYear: string): StudentSemesterInfo | null {
    return getStudentCurrentSemester(enrollmentYear);
  }

  static getSemesterDescription(enrollmentYear?: string): string {
    return getSemesterDescription(enrollmentYear);
  }

  static getGradeName(semesterNumber: number): string {
    return getGradeName(semesterNumber);
  }

  static getLatestSemesterFromGrades(grades: GradeWithSemester[]): SemesterInfo | null {
    return getLatestSemesterFromGrades(grades);
  }

  static inferEnrollmentYearFromGrades(grades: GradeWithSemester[]): string | null {
    return inferEnrollmentYearFromGrades(grades);
  }

  static getEarliestSemesterFromGrades(grades: GradeWithSemester[]): SemesterInfo | null {
    return getEarliestSemesterFromGrades(grades);
  }

  static getSemesterCount(grades: GradeWithSemester[]): number {
    return getSemesterCount(grades);
  }
}
