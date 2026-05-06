/**
 * 关于我们页面模块
 * 
 * 展示应用相关信息，包括：
 * - 应用简介
 * - 主要功能介绍
 * - 联系方式
 * - 版本信息
 * 
 * @module AboutPage
 */

import { router } from '@kit.ArkUI';

/**
 * 关于我们页面组件
 * 
 * 展示应用的详细信息
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/AboutPage' });
 * ```
 */
@Entry
@Component
struct AboutPage {
  build() {
    Column() {
      Column()
        .width('100%')
        .height(44)
      this.buildHeader()
      this.buildContent()
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_quaternary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
  }

  /**
   * 构建页面头部
   * 包含返回按钮和标题
   */
  @Builder
  buildHeader() {
    Row() {
      Image($r('app.media.chevron_left'))
        .width(24)
        .height(24)
        .fillColor($r('app.color.text_primary'))
        .onClick(() => {
          router.back();
        })
      Text('关于我们')
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
        .textAlign(TextAlign.Center)
      Blank()
        .width(24)
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    .backgroundColor($r('app.color.background_primary'))
  }

  /**
   * 构建页面内容
   * 包含应用简介、功能、联系方式和版本信息
   */
  @Builder
  buildContent() {
    Scroll() {
      Column({ space: 16 }) {
        this.buildAppDescription()
        this.buildDashedDivider()
        this.buildFeatures()
        this.buildDashedDivider()
        this.buildContact()
        this.buildDashedDivider()
        this.buildVersion()
      }
      .width('100%')
      .padding(16)
    }
    .layoutWeight(1)
  }

  /**
   * 构建应用简介卡片
   */
  @Builder
  buildAppDescription() {
    Column() {
      Text('应用简介')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Text('青序是一款专为学生打造的综合性校园服务应用。我们致力于通过数字化手段，为同学们提供便捷、高效的校园生活体验。')
        .fontSize(14)
        .fontColor($r('app.color.text_tertiary'))
        .lineHeight(22)
        .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  /**
   * 构建主要功能卡片
   */
  @Builder
  buildFeatures() {
    Column() {
      Text('主要功能')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 12 }) {
        this.buildFeatureItem('课程表', '查看个人课程安排，支持周数切换和提醒')
        this.buildFeatureItem('成绩查询', '快速查询考试成绩，了解学业表现')
        this.buildFeatureItem('培养计划', '查看专业培养方案，规划学习路径')
        this.buildFeatureItem('学分进度', '查看学分完成情况，掌握学业进度')
        this.buildFeatureItem('考试安排', '查看考试时间和地点，合理安排复习')
      }
      .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  /**
   * 构建功能项
   * 
   * @param title - 功能名称
   * @param desc - 功能描述
   */
  @Builder
  buildFeatureItem(title: string, desc: string) {
    Row() {
      Text(title)
        .fontSize(14)
        .fontColor($r('app.color.text_primary'))
        .fontWeight(FontWeight.Medium)
        .width('40%')

      Text(desc)
        .fontSize(13)
        .fontColor($r('app.color.text_tertiary'))
        .layoutWeight(1)
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
  }

  /**
   * 构建联系方式卡片
   */
  @Builder
  buildContact() {
    Column() {
      Text('联系我们')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildContactItem('邮箱', 'yelan12192023@163.com')
      }
      .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  /**
   * 构建联系项
   * 
   * @param title - 联系方式名称
   * @param value - 联系方式值
   */
  @Builder
  buildContactItem(title: string, value: string) {
    Row() {
      Text(title)
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .width('40%')

      Text(value)
        .fontSize(14)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
    }
    .width('100%')
  }

  /**
   * 构建版本信息卡片
   */
  @Builder
  buildVersion() {
    Column() {
      Text('版本信息')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildVersionItem('当前版本', 'v1.0.0')
      }
      .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  /**
   * 构建版本项
   * 
   * @param title - 版本项名称
   * @param value - 版本项值
   */
  @Builder
  buildVersionItem(title: string, value: string) {
    Row() {
      Text(title)
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .width('40%')

      Text(value)
        .fontSize(14)
        .fontColor($r('app.color.text_primary'))
        .layoutWeight(1)
    }
    .width('100%')
  }

  /**
   * 构建虚线分隔符
   */
  @Builder
  buildDashedDivider() {
    Row() {
      ForEach([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], (index: number) => {
        if (index % 2 === 0) {
          Column()
            .width(8)
            .height(1)
            .backgroundColor($r('app.color.divider'))
        } else {
          Column()
            .width(4)
            .height(1)
        }
      })
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
  }
}

export { AboutPage };
