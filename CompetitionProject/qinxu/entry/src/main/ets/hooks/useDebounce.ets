/**
 * 防抖Hook模块
 * 
 * 提供函数防抖功能，用于延迟执行频繁触发的操作
 * 
 * @module useDebounce
 */

import { AppConstants } from '../common/constants/AppConstants';

/**
 * 防抖函数Hook
 * 
 * 创建一个防抖函数，在指定延迟时间内只执行最后一次调用
 * 适用于搜索输入、窗口调整等频繁触发场景
 * 
 * @template T - 函数类型
 * @param fn - 需要防抖的函数
 * @param delay - 延迟时间（毫秒），默认使用AppConstants.TIME.DEBOUNCE_DELAY
 * @returns 防抖后的函数
 * 
 * @example
 * ```typescript
 * const debouncedSearch = useDebounce((keyword: string) => {
 *   console.log('搜索:', keyword);
 * }, 300);
 * 
 * // 快速调用多次，只会在最后一次调用后300ms执行
 * debouncedSearch('test');
 * debouncedSearch('testing');
 * debouncedSearch('testing now'); // 只有这次会执行
 * ```
 */
export function useDebounce<T extends (...args: Object[]) => void>(
  fn: T,
  delay: number = AppConstants.TIME.DEBOUNCE_DELAY
): T {
  /** 定时器ID */
  let timer: number = -1;
  
  return ((...args: Object[]) => {
    if (timer !== -1) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      fn(...args);
      timer = -1;
    }, delay);
  }) as T;
}
