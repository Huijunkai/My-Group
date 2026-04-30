/**
 * 通用类型定义模块
 * 
 * 定义应用通用的数据类型，包括：
 * - 主题配置类型
 * - 导航状态类型
 * - 加载状态类型
 * - 错误状态类型
 * - API响应类型
 * 
 * @module common
 */

/**
 * 主题配置接口
 */
export interface IThemeConfig {
  /** 主题模式 */
  mode: 'light' | 'dark' | 'system';
  /** 主色调 */
  primaryColor: string;
  /** 背景色 */
  backgroundColor: string;
  /** 文字颜色 */
  textColor: string;
}

/**
 * 导航状态接口
 */
export interface INavigationState {
  /** 当前页面 */
  currentPage: string;
  /** 上一个页面 */
  previousPage: string | null;
  /** 导航参数 */
  params: Record<string, Object>;
}

/**
 * 加载状态接口
 */
export interface ILoadingState {
  /** 是否加载中 */
  isLoading: boolean;
  /** 加载提示消息 */
  message: string;
}

/**
 * 错误状态接口
 */
export interface IErrorState {
  /** 是否有错误 */
  hasError: boolean;
  /** 错误消息 */
  message: string;
  /** 错误代码 */
  code?: string;
}

/**
 * API响应接口
 * 
 * @template T - 响应数据类型
 */
export interface IApiResponse<T> {
  /** 是否成功 */
  success: boolean;
  /** 响应数据 */
  data?: T;
  /** 响应消息 */
  message?: string;
  /** 状态码 */
  code?: number;
}
