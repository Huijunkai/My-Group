/**
 * 节流Hook模块
 * 
 * 提供函数节流功能，用于限制函数的执行频率
 * 
 * @module useThrottle
 */

import { AppConstants } from '../common/constants/AppConstants';

/**
 * 节流函数Hook
 * 
 * 创建一个节流函数，在指定时间间隔内只执行一次
 * 适用于滚动事件、按钮点击等需要限制频率的场景
 * 
 * @template T - 函数类型
 * @param fn - 需要节流的函数
 * @param delay - 执行间隔（毫秒），默认使用AppConstants.TIME.THROTTLE_DELAY
 * @returns 节流后的函数
 * 
 * @example
 * ```typescript
 * const throttledScroll = useThrottle((position: number) => {
 *   console.log('滚动位置:', position);
 * }, 100);
 * 
 * // 快速调用多次，每100ms最多执行一次
 * throttledScroll(100);
 * throttledScroll(200);
 * throttledScroll(300); // 只有这次会执行（假设在100ms内）
 * ```
 */
export function useThrottle<T extends (...args: Object[]) => void>(
  fn: T,
  delay: number = AppConstants.TIME.THROTTLE_DELAY
): T {
  /** 上次执行时间戳 */
  let lastTime: number = 0;
  
  return ((...args: Object[]) => {
    const now = Date.now();
    if (now - lastTime >= delay) {
      fn(...args);
      lastTime = now;
    }
  }) as T;
}
