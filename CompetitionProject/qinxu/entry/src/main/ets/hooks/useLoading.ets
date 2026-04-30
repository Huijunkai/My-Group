/**
 * 加载状态Hook模块
 * 
 * 提供加载状态管理功能，用于控制加载提示的显示和隐藏
 * 
 * @module useLoading
 */

/**
 * 加载控制器接口
 */
export interface ILoadingController {
  /** 显示加载提示
   * @param message - 加载提示信息，默认为'加载中...'
   */
  show: (message?: string) => void;
  /** 隐藏加载提示 */
  hide: () => void;
  /** 获取当前加载状态
   * @returns 是否正在加载中
   */
  isLoading: () => boolean;
}

/**
 * 加载状态Hook
 * 
 * 创建一个加载状态控制器，用于管理加载提示的显示状态
 * 
 * @returns 加载控制器对象
 * 
 * @example
 * ```typescript
 * const loading = useLoading();
 * 
 * // 显示加载
 * loading.show('正在获取数据...');
 * 
 * // 检查状态
 * if (loading.isLoading()) {
 *   console.log('正在加载中');
 * }
 * 
 * // 隐藏加载
 * loading.hide();
 * ```
 */
export function useLoading(): ILoadingController {
  /** 加载状态 */
  let loadingState: boolean = false;
  /** 加载提示信息 */
  let loadingMessage: string = '';
  
  return {
    show: (message?: string) => {
      loadingState = true;
      loadingMessage = message || '加载中...';
    },
    hide: () => {
      loadingState = false;
      loadingMessage = '';
    },
    isLoading: () => loadingState
  };
}
