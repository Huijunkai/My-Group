/**
 * 常用功能页面模块
 * 
 * 提供常用功能快捷入口，包括：
 * - 成绩查询
 * - 培养计划
 * - 学分进度
 * - 考试安排
 * 
 * @module CommonPage
 */

import { router } from '@kit.ArkUI';

/**
 * 常用功能页面组件
 * 
 * 展示常用功能列表入口
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/CommonPage' });
 * ```
 */
@Entry
@Component
struct CommonPage {
  build() {
    Column() {
      // 顶部导航栏
      Row() {
        Image($r('app.media.chevron_left'))
          .width(24)
          .height(24)
          .fillColor($r('app.color.text_primary'))
          .onClick(() => {
            router.back();
          })
        Text('常用功能')
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .layoutWeight(1)
          .textAlign(TextAlign.Center)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor($r('app.color.background_primary'))

      // 常用功能列表
      Scroll() {
        Column({ space: 12 }) {
          this.buildFunctionItem('成绩查询', '查看考试成绩', $r('app.media.checkmark_clipboard_fill'))
          this.buildFunctionItem('培养计划', '查看专业培养方案', $r('app.media.doc_text_badge_checkmark'))
          this.buildFunctionItem('学分进度', '查看学分完成情况', $r('app.media.doc_plaintext_and_pencil_fill'))
          this.buildFunctionItem('考试安排', '查看考试安排', $r('app.media.text_clipboard'))
        }
        .width('100%')
        .padding(16)
      }
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
  }

  /**
   * 构建功能项
   * 
   * @param title - 功能名称
   * @param desc - 功能描述
   * @param icon - 图标资源
   */
  @Builder
  buildFunctionItem(title: string, desc: string, icon: Resource) {
    Row() {
      Image(icon)
        .width(24)
        .height(24)
        .margin({ right: 12 })
      
      Column({ space: 2 }) {
        Text(title)
          .fontSize(16)
          .fontColor($r('app.color.text_primary'))
        Text(desc)
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
      }
      .layoutWeight(1)

      Image($r('app.media.chevron_right'))
        .width(16)
        .height(16)
        .fillColor($r('app.color.text_tertiary'))
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(8)
    .onClick(() => {
      switch (title) {
        case '成绩查询':
          router.pushUrl({ url: 'pages/GradeQueryPage' });
          break;
        case '培养计划':
          router.pushUrl({ url: 'pages/TrainingPlanPage' });
          break;
        case '学分进度':
          router.pushUrl({ url: 'pages/CreditProgressPage' });
          break;
        case '考试安排':
          router.pushUrl({ url: 'pages/ExamPage' });
          break;
      }
    })
  }
}

export { CommonPage };