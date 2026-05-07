/**
 * 隐私设置页面模块
 * 
 * 提供隐私相关设置功能，包括：
 * - 隐私说明
 * - 隐私政策查看
 * 
 * @module PrivacySettingsPage
 */

import { router } from '@kit.ArkUI';

/**
 * 隐私设置页面组件
 * 
 * 展示和管理隐私相关设置
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/PrivacySettingsPage' });
 * ```
 */
@Entry
@Component
struct PrivacySettingsPage {

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
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
  }

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
      Text('隐私设置')
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
  }

  @Builder
  buildContent() {
    Scroll() {
      Column({ space: 16 }) {
        this.buildPrivacyIntro()
        this.buildDashedDivider()
        this.buildPrivacyPolicy()
      }
      .width('100%')
      .padding(16)
    }
    .layoutWeight(1)
  }

  @Builder
  buildPrivacyIntro() {
    Column() {
      Text('隐私保护说明')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Text('青序 APP 系由个人开发者独立开发的非官方移动应用程序。本应用仅通过教务系统采集您的必要数据以提供服务，用户个人信息将采取本地存储与云端加密存储相结合的方式。本地存储用于快速访问和离线使用，云端存储用于数据备份和跨设备同步。所有云端存储的个人数据均经过严格的加密处理，采用AES-256位加密算法进行数据加密，确保数据在传输和存储过程中的安全性。')
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .lineHeight(22)
        .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildPrivacyPolicy() {
    Column() {
      Row() {
        Text('隐私政策与软件说明')
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))
          .fontWeight(FontWeight.Medium)
          .layoutWeight(1)

        Image($r('app.media.chevron_right'))
          .width(14)
          .height(14)
          .fillColor($r('app.color.text_tertiary'))
      }
      .width('100%')
      .height(52)
      .onClick(() => {
        this.showPrivacyPolicy();
      })
    }
    .width('100%')
    .padding({ left: 16, right: 16 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

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
      }, (index: number) => `divider_${index}`)
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
  }

  private showPrivacyPolicy(): void {
    router.pushUrl({ url: 'pages/PrivacyPolicyPage' });
  }
}

export { PrivacySettingsPage };
