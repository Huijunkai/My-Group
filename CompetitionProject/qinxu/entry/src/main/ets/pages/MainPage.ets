/**
 * 主页面模块
 * 
 * 应用的主入口页面，包含：
 * - 底部导航栏
 * - 页面切换管理
 * - 返回键处理
 * 
 * @module MainPage
 */

import { GlassMenubar } from '../components/GlassMenubar';

/**
 * 主页面组件
 * 
 * 作为应用的主容器，承载底部导航和各功能页面
 * 
 * @example
 * ```typescript
 * router.replaceUrl({ url: 'pages/MainPage' });
 * ```
 */
@Entry
@Component
struct MainPage {
  /** 显示公告详情标志（全局存储链接） */
  @StorageLink('showAnnouncementDetail') showAnnouncementDetail: boolean = false;

  /**
   * 返回键按下回调
   * 处理公告详情的关闭
   * 
   * @returns 是否拦截返回键
   */
  onBackPress(): boolean {
    if (this.showAnnouncementDetail) {
      AppStorage.SetOrCreate('showAnnouncementDetail', false);
      return true;
    }
    return false;
  }

  build() {
    Column() {
      GlassMenubar()
    }
    .width('100%')
    .height('100%')
  }
}
