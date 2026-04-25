import { http } from '@kit.NetworkKit';

/**
 * 节假日信息接口
 */
export interface HolidayInfo {
  name: string;
  isWorkday: boolean;
}

/**
 * 农历日期信息接口
 */
export interface LunarDateInfo {
  year: number;
  month: number;
  day: number;
  isLeap: boolean;
}

/**
 * 农历工具类
 * 提供农历日期转换和法定节假日查询功能
 */
export class LunarCalendar {
  // 农历月份名称
  private static readonly MONTHS: string[] = [
    '正月', '二月', '三月', '四月', '五月', '六月',
    '七月', '八月', '九月', '十月', '冬月', '腊月'
  ];

  // 农历日期名称
  private static readonly DAYS: string[] = [
    '初一', '初二', '初三', '初四', '初五', '初六', '初七', '初八', '初九', '初十',
    '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十',
    '廿一', '廿二', '廿三', '廿四', '廿五', '廿六', '廿七', '廿八', '廿九', '三十'
  ];

  // 生肖
  private static readonly ZODIACS: string[] = [
    '鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪'
  ];

  // 天干
  private static readonly HEAVENLY_STEMS: string[] = [
    '甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'
  ];

  // 地支
  private static readonly EARTHLY_BRANCHES: string[] = [
    '子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'
  ];

  // 农历数据表（1900-2100）
  private static readonly LUNAR_INFO: number[] = [
    0x04bd8, 0x04ae0, 0x0a570, 0x054d5, 0x0d260, 0x0d950, 0x16554, 0x056a0, 0x09ad0, 0x055d2,
    0x04ae0, 0x0a5b6, 0x0a4d0, 0x0d250, 0x1d255, 0x0b540, 0x0d6a0, 0x0ada2, 0x095b0, 0x14977,
    0x04970, 0x0a4b0, 0x0b4b5, 0x06a50, 0x06d40, 0x1ab54, 0x02b60, 0x09570, 0x052f2, 0x04970,
    0x06566, 0x0d4a0, 0x0ea50, 0x06e95, 0x05ad0, 0x02b60, 0x186e3, 0x092e0, 0x1c8d7, 0x0c950,
    0x0d4a0, 0x1d8a6, 0x0b550, 0x056a0, 0x1a5b4, 0x025d0, 0x092d0, 0x0d2b2, 0x0a950, 0x0b557,
    0x06ca0, 0x0b550, 0x15355, 0x04da0, 0x0a5b0, 0x14573, 0x052b0, 0x0a9a8, 0x0e950, 0x06aa0,
    0x0aea6, 0x0ab50, 0x04b60, 0x0aae4, 0x0a570, 0x05260, 0x0f263, 0x0d950, 0x05b57, 0x056a0,
    0x096d0, 0x04dd5, 0x04ad0, 0x0a4d0, 0x0d4d4, 0x0d250, 0x0d558, 0x0b540, 0x0b6a0, 0x195a6,
    0x095b0, 0x049b0, 0x0a974, 0x0a4b0, 0x0b27a, 0x06a50, 0x06d40, 0x0af46, 0x0ab60, 0x09570,
    0x04af5, 0x04970, 0x064b0, 0x074a3, 0x0ea50, 0x06b58, 0x055c0, 0x0ab60, 0x096d5, 0x092e0,
    0x0c960, 0x0d954, 0x0d4a0, 0x0da50, 0x07552, 0x056a0, 0x0abb7, 0x025d0, 0x092d0, 0x0cab5,
    0x0a950, 0x0b4a0, 0x0baa4, 0x0ad50, 0x055d9, 0x04ba0, 0x0a5b0, 0x15176, 0x052b0, 0x0a930,
    0x07954, 0x06aa0, 0x0ad50, 0x05b52, 0x04b60, 0x0a6e6, 0x0a4e0, 0x0d260, 0x0ea65, 0x0d530,
    0x05aa0, 0x076a3, 0x096d0, 0x04bd7, 0x04ad0, 0x0a4d0, 0x1d0b6, 0x0d250, 0x0d520, 0x0dd45,
    0x0b5a0, 0x056d0, 0x055b2, 0x049b0, 0x0a577, 0x0a4b0, 0x0aa50, 0x1b255, 0x06d20, 0x0ada0,
    0x14b63, 0x09370, 0x049f8, 0x04970, 0x064b0, 0x168a6, 0x0ea50, 0x06b20, 0x1a6c4, 0x0aae0,
    0x0a2e0, 0x0d2e3, 0x0c960, 0x0d557, 0x0d4a0, 0x0da50, 0x05d55, 0x056a0, 0x0a6d0, 0x055d4,
    0x052d0, 0x0a9b8, 0x0a950, 0x0b4a0, 0x0b6a6, 0x0ad50, 0x055a0, 0x0aba4, 0x0a5b0, 0x052b0,
    0x0b273, 0x06930, 0x07337, 0x06aa0, 0x0ad50, 0x14b55, 0x04b60, 0x0a570, 0x054e4, 0x0d160,
    0x0e968, 0x0d520, 0x0daa0, 0x16aa6, 0x056d0, 0x04ae0, 0x0a9d4, 0x0a2d0, 0x0d150, 0x0f252,
    0x0d520
  ];

  /**
   * 公历转农历
   * @param date 公历日期
   * @returns 农历日期信息
   */
  static solarToLunar(date: Date): LunarDateInfo {
    const baseDate = new Date(1900, 0, 31);
    const year = date.getFullYear();
    const month = date.getMonth();
    const day = date.getDate();
    
    const baseTime = Date.UTC(baseDate.getFullYear(), baseDate.getMonth(), baseDate.getDate());
    const inputTime = Date.UTC(year, month, day);
    let offset = Math.floor((inputTime - baseTime) / 86400000);

    let lunarYear = 1900;
    let lunarMonth = 1;
    let lunarDay = 1;
    let isLeap = false;

    // 计算农历年份
    for (lunarYear = 1900; lunarYear < 2101; lunarYear++) {
      const yearDays = LunarCalendar.getYearDays(lunarYear);
      if (offset < yearDays) {
        break;
      }
      offset -= yearDays;
    }

    // 计算农历月份和日期
    const leapMonth = LunarCalendar.getLeapMonth(lunarYear);
    let leapMonthDays = 0;
    if (leapMonth !== 0) {
      leapMonthDays = LunarCalendar.getLeapMonthDays(lunarYear);
    }

    let monthDays = 0;
    for (lunarMonth = 1; lunarMonth <= 12; lunarMonth++) {
      // 处理闰月：当存在闰月且当前月份到达闰月位置时，先处理闰月
      if (leapMonth > 0 && lunarMonth === (leapMonth + 1) && !isLeap) {
        --lunarMonth;
        isLeap = true;
        monthDays = leapMonthDays;
      } else {
        monthDays = LunarCalendar.getMonthDays(lunarYear, lunarMonth);
      }

      if (isLeap && monthDays === leapMonthDays) {
        isLeap = false;
      }

      if (offset < monthDays) {
        break;
      }
      offset -= monthDays;
    }

    if (offset === 0 && leapMonth > 0 && lunarMonth === leapMonth + 1) {
      if (isLeap) {
        isLeap = false;
      } else {
        isLeap = true;
        --lunarMonth;
      }
    }

    if (offset < 0) {
      offset += monthDays;
      --lunarMonth;
      lunarDay = offset + 1;
    } else {
      lunarDay = offset + 1;
    }

    if (lunarMonth > 12) {
      lunarMonth = 12;
    }
    if (lunarMonth < 1) {
      lunarMonth = 1;
    }

    return { year: lunarYear, month: lunarMonth, day: lunarDay, isLeap };
  }

  /**
   * 获取农历年份的总天数
   */
  private static getYearDays(year: number): number {
    let sum = 348;
    const info = LunarCalendar.LUNAR_INFO[year - 1900];
    for (let i = 0x8000; i > 0x8; i >>= 1) {
      sum += (info & i) ? 1 : 0;
    }
    return sum + LunarCalendar.getLeapMonthDays(year);
  }

  /**
   * 获取农历月份的天数
   */
  private static getMonthDays(year: number, month: number): number {
    if (month < 1 || month > 12) {
      return 29;
    }
    const info = LunarCalendar.LUNAR_INFO[year - 1900];
    return (info & (0x10000 >> (month - 1))) ? 30 : 29;
  }

  /**
   * 获取闰月天数
   */
  private static getLeapMonthDays(year: number): number {
    const leap = LunarCalendar.getLeapMonth(year);
    if (leap !== 0) {
      const info = LunarCalendar.LUNAR_INFO[year - 1900];
      return (info & 0x10000) ? 30 : 29;
    }
    return 0;
  }

  /**
   * 获取闰月月份
   */
  private static getLeapMonth(year: number): number {
    if (year < 1900 || year > 2100) {
      return 0;
    }
    const info = LunarCalendar.LUNAR_INFO[year - 1900];
    return info & 0xf;
  }

  /**
   * 获取生肖
   */
  static getZodiac(year: number): string {
    // 生肖以立春为界，2026 年 2 月 4 日立春后才是丙午年
    // 简化计算：以春节为界，2026 年春节是 1 月 29 日
    const baseYear = 1900;
    const offset = year - baseYear;
    return LunarCalendar.ZODIACS[offset % 12];
  }

  /**
   * 获取天干地支纪年
   */
  static getGanZhiYear(year: number): string {
    // 天干地支以立春为界，但这里简化处理，以春节为界
    const stemIndex = (year - 4) % 10;
    const branchIndex = (year - 4) % 12;
    return LunarCalendar.HEAVENLY_STEMS[stemIndex] + LunarCalendar.EARTHLY_BRANCHES[branchIndex] + '年';
  }

  /**
   * 获取农历月份名称
   */
  static getLunarMonthName(month: number, isLeap: boolean): string {
    if (month < 1 || month > 12) {
      return '';
    }
    return (isLeap ? '闰' : '') + LunarCalendar.MONTHS[month - 1];
  }

  /**
   * 获取农历日期名称
   */
  static getLunarDayName(day: number): string {
    if (day < 1 || day > 30) {
      return '';
    }
    return LunarCalendar.DAYS[day - 1];
  }

  /**
   * 获取完整的农历日期字符串
   * @param date 公历日期
   * @returns 格式化的农历字符串
   */
  static getLunarDateString(date: Date): string {
    const lunar = LunarCalendar.solarToLunar(date);
    
    // 使用农历年份来计算生肖和天干地支
    // 确保干支/生肖与农历月日一致（春节前公历已进新年但农历仍在旧年）
    const ganZhi = LunarCalendar.getGanZhiYear(lunar.year);
    const zodiac = LunarCalendar.getZodiac(lunar.year);
    const monthName = LunarCalendar.getLunarMonthName(lunar.month, lunar.isLeap);
    const dayName = LunarCalendar.getLunarDayName(lunar.day);

    if (!monthName || !dayName) {
      return '农历计算中...';
    }

    return `${ganZhi}（${zodiac}）${monthName}${dayName}`;
  }

  /**
   * 从 API 获取法定节假日数据
   * @param year 年份
   * @returns 节假日数据
   */
  static async getHolidays(year: number): Promise<Map<string, HolidayInfo>> {
    const holidays: Map<string, HolidayInfo> = new Map();

    try {
      // 使用中国节假日 API
      const url = `https://api.qingyunke.com/api.php?key=free&appid=01&msg=${year}年节假日`;

      const httpRequest = http.createHttp();
      const response = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 5000,
        readTimeout: 5000
      });

      if (response.responseCode === 200) {
        const data: Record<string, Object> = JSON.parse(response.result as string);
        // 解析节假日数据
        if (data && data.holidays) {
          const holidaysArray = data.holidays as Array<Record<string, Object>>;
          for (const holiday of holidaysArray) {
            const date = holiday.date as string;
            const name = holiday.name as string;
            const isWorkday = (holiday.isWorkday as boolean) || false;
            holidays.set(date, {
              name: name,
              isWorkday: isWorkday
            });
          }
        }
      }
    } catch (error) {
      console.error('LunarCalendar: Failed to fetch holidays:', error);
    }

    return holidays;
  }

  /**
   * 检查日期是否为法定节假日
   * @param date 日期
   * @param holidays 节假日数据
   * @returns 是否为节假日
   */
  static isHoliday(date: Date, holidays: Map<string, HolidayInfo>): boolean {
    const dateStr = LunarCalendar.formatDate(date);
    return holidays.has(dateStr);
  }

  /**
   * 检查日期是否为调休工作日
   * @param date 日期
   * @param holidays 节假日数据
   * @returns 是否为调休工作日
   */
  static isWorkday(date: Date, holidays: Map<string, HolidayInfo>): boolean {
    const dateStr = LunarCalendar.formatDate(date);
    const holiday = holidays.get(dateStr);
    return holiday !== undefined && holiday.isWorkday;
  }

  /**
   * 格式化日期为 YYYY-MM-DD
   */
  private static formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  /**
   * 获取节假日名称
   * @param date 日期
   * @param holidays 节假日数据
   * @returns 节假日名称
   */
  static getHolidayName(date: Date, holidays: Map<string, HolidayInfo>): string {
    const dateStr = LunarCalendar.formatDate(date);
    const holiday = holidays.get(dateStr);
    return holiday ? holiday.name : '';
  }
}
