/**
 * API 常量配置模块
 * 
 * 定义应用中使用的 API 相关常量，包括：
 * - 基础 URL 地址
 * - API 端点路径
 * - URL 构建工具方法
 * 
 * @module ApiConstants
 */

/**
 * API 常量类
 * 
 * 提供统一的 API 配置管理
 * 
 * @example
 * ```typescript
 * const url = ApiConstants.getFullUrl(ApiConstants.ENDPOINTS.SYNC);
 * // 结果: 'http://124.70.92.199:3001/api/sync'
 * ```
 */
export class ApiConstants {
  /** API 基础 URL 地址 */
  static readonly BASE_URL: string = 'http://124.70.92.199:3001';
  
  /**
   * API 端点路径常量
   * 
   * @property SYNC - 数据同步端点
   * @property STUDENT_DATA - 学生数据端点
   * @property LATEST_SEMESTER - 最新学期信息端点
   */
  static readonly ENDPOINTS = {
    SYNC: '/api/sync',
    STUDENT_DATA: '/api/student',
    LATEST_SEMESTER: '/api/semester/latest'
  } as const;
  
  /**
   * 获取完整的 API URL
   * 
   * 将基础 URL 和端点路径组合成完整的 URL
   * 
   * @param endpoint - API 端点路径
   * @returns 完整的 API URL
   */
  static getFullUrl(endpoint: string): string {
    return `${ApiConstants.BASE_URL}${endpoint}`;
  }
}
