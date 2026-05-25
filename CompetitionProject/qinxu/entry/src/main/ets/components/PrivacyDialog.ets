/**
 * 隐私政策对话框组件模块
 * 
 * 提供用户协议与隐私政策展示，包括：
 * - 隐私政策内容展示
 * - 滚动阅读检测
 * - 同意/不同意操作
 * 
 * @module PrivacyDialog
 */

import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'privacy_preferences';
/** 隐私协议同意状态键名 */
const KEY_PRIVACY_AGREED = 'privacyAgreed';

/**
 * 隐私政策对话框组件
 * 
 * 展示用户协议与隐私政策，需要用户滚动阅读完毕后才能同意
 * 
 * @example
 * ```typescript
 * PrivacyDialog({
 *   isVisible: $isVisible,
 *   onAgreed: () => { console.log('用户已同意'); }
 * })
 * ```
 */
@Component
export struct PrivacyDialog {
  /** 对话框可见状态（双向绑定） */
  @Link isVisible: boolean;
  /** 是否已滚动到底部 */
  @State hasScrolledToBottom: boolean = false;
  /** 同意回调函数 */
  onAgreed: () => void = () => {};
  /** 滚动控制器 */
  private scroller: Scroller = new Scroller();
  /** 应用上下文 */
  private context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;

  build() {
    if (this.isVisible) {
      Column() {
        Column() {
          Text('用户协议与隐私政策')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))
            .margin({ top: 24, bottom: 16 })

          Column() {
            Scroll(this.scroller) {
              Column() {
                Text('软件说明')
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 12 })

                Text('青序 APP（以下简称"本应用"）系由个人开发者独立开发并运营的非官方移动应用程序。本应用与任何学校、教育机构之间不存在任何形式的合作关系、授权关系，亦未签署数据对接协议或达成合作意向。用户通过自主下载、注册、登录及使用本应用的行为，均属于个人自愿行为，与学校要求或委托无关。特此声明，学校对本应用的功能完整性、系统安全性、数据存储合规性、服务稳定性等方面不承担任何法律责任及义务，用户在使用过程中产生的一切风险需自行承担。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 12 })

                Text('本应用严格遵循标识使用相关规范，在产品全生命周期管理过程中，包含产品设计、开发及运营等各环节，均未使用任何学校的名称、LOGO、徽章等专属标识。在市场推广层面，无论是广告文案、新闻稿件，还是其他宣传资料，均杜绝通过文字表述制造与学校存在关联的误导性信息；在用户界面设计方面，所有界面元素、图标样式均与学校视觉识别系统保持明确区分；功能模块命名亦规避使用易引发联想的词汇。此外，应用开发团队已构建完善的内部审核体系，针对新增功能、宣传物料、运营活动等内容开展多轮严格审核，确保不出现任何暗示与学校存在关联的信息，切实维护用户的知情权，保障市场竞争秩序的规范性与公正性。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 20 })

                Text('使用前提')
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 12 })

                Text('用户点击"登录/使用"按钮即表示：')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('1. 用户已详细阅读、充分理解并自愿接受本隐私政策及软件说明的全部条款，包括但不限于个人信息收集、使用、存储、共享等相关规定。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 4 })

                Text('2. 用户同意授权本应用在合理必要范围内，收集设备信息（如 IMEI 号、设备型号、操作系统版本）及网络信息（IP 地址），以确保服务的稳定性与安全性。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 4 })

                Text('3. 用户知悉并确认，本应用将依据功能需求，收集地理位置、通知权限数据。所有收集的数据仅用于实现对应服务功能，绝不作超范围使用。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 4 })

                Text('4. 用户理解本隐私政策可能因法律法规变更或业务发展需要进行修订，并同意定期查阅更新内容。继续使用本应用即视为用户接受修订后的条款。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 4 })

                Text('5. 用户明确知晓，本 APP 为个人独立开发的非官方产品。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 4 })

                Text('6. 用户同意本隐私政策中关于个人信息收集、存储及使用的所有条款。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 20 })

                Text('隐私政策')
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 12 })

                Text('（一）信息收集与使用')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 8 })

                Text('1. 信息收集范围：为保障本应用系统的安全稳定运行，依据相关法律法规要求，本平台将收集用户主动提交的个人信息。具体涵盖您的用户凭据（学号，密码），账户注册资料（如学生真实姓名，性别）、学习相关信息（个人成绩、培养计划、学分获取进度、课程信息、入学时间、专业和学院名称）、设备识别标识等数据。其中部分信息属于敏感个人信息范畴，涉及用户个人隐私。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('2. 信息收集准则：本平台严格恪守《中华人民共和国个人信息保护法》《网络数据安全管理条例》等国家法律法规规定，遵循最小必要原则开展信息收集工作。仅采集实现产品服务功能所必需的个人信息，坚决杜绝超范围收集与产品服务无关的用户数据。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('3. 信息使用目的：本平台所收集的个人信息，仅用于优化产品功能与服务体验、强化账户安全防护体系，以及响应处理用户服务需求。未经用户明确授权同意，本平台承诺不会将用户个人信息用于任何形式的商业推广或第三方营销活动。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('4. 使用用户凭据：您的用户凭据（学号，密码），只会向教务系统发起登录请求，出现的保存密码以及记住密码只会保存在您的设备上。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 16 })

                Text('（二）信息存储与保护')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 8 })

                Text('1. 存储方式：用户个人信息将采取本地存储与云端加密存储相结合的方式。本地存储用于快速访问和离线使用，云端存储用于数据备份和跨设备同步。所有云端存储的个人数据均经过严格的加密处理，采用AES-256位加密算法进行数据加密，确保数据在传输和存储过程中的安全性。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('2. 加密机制：云端存储的数据在传输过程中采用TLS/SSL协议进行加密，存储时采用行业标准的加密算法和密钥管理机制。加密密钥严格保管，定期轮换，确保即使在极端情况下数据也无法被未授权访问。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('3. 存储期限：用户个人信息的存储周期严格遵循数据最小化原则，仅在实现服务目标所需的最短时间内留存。用户完成账号注销操作后，将立即对本地和云端存储的相关信息进行彻底删除，确保数据隐私安全。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('4. 安全措施：本平台实施多层次安全防护措施，包括但不限于：访问控制机制、异常登录监测、数据传输加密、定期安全审计等，全方位保障用户数据安全。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 16 })

                Text('（三）用户权利')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 8 })

                Text('1. 用户享有对其个人信息的查阅、复制、更正、补充及删除权利。用户可通过本 APP 内置反馈渠道，或经由开发者公示的官方联系方式提交相关申请。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('2. 用户有权随时撤回对个人信息收集与使用的授权同意。需特别说明的是，撤回非必要信息授权不会影响核心服务功能的正常使用；若撤回必要信息授权，可能导致部分功能无法正常运行。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 16 })

                Text('（四）免责声明')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 8 })

                Text('1. 本 APP 开发者作为个人信息处理者，严格依据相关法律法规履行数据安全保护义务。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('2. 针对用户使用本 APP 过程中产生的一切争议（涵盖但不限于账号异常、服务质量争议、数据安全纠纷等），均由用户与开发者基于相关法律法规及本隐私政策约定协商解决。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('3. 开发者仅对本 APP 功能设计及运营的合法性负责。用户利用本 APP 实施的任何违法违规行为，其法律后果及责任均由用户自行承担，开发者不对此类行为产生的后果承担法律责任。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 16 })

                Text('（五）其他条款')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .width('100%')
                  .margin({ bottom: 8 })

                Text('1. 若对本隐私政策进行修订，我们将通过 APP 弹窗、站内公告等显著方式向用户进行公示。用户继续使用本应用，即表明已充分知悉并同意修订后的隐私政策条款内容。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('2. 如用户有任何咨询、投诉或权利主张需求，可通过以下方式与我们取得联系：yelan12192023@163.com。我们将及时响应并处理用户反馈。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 8 })

                Text('3. 本隐私政策及软件说明的最终解释权归本 APP 开发主体所有。未尽事宜，将严格依照国家相关法律法规及监管要求执行。')
                  .fontSize(13)
                  .fontColor($r('app.color.text_secondary'))
                  .width('100%')
                  .lineHeight(22)
                  .margin({ bottom: 40 })
              }
              .width('100%')
              .padding({ left: 20, right: 20 })
            }
            .width('100%')
            .height(300)
            .scrollBar(BarState.Auto)
            .scrollBarColor($r('app.color.text_tertiary'))
            .scrollBarWidth(4)
            .edgeEffect(EdgeEffect.Spring)
            .onScroll(() => {
              this.checkScrollPosition();
            })
            .onScrollEnd(() => {
              this.checkScrollPosition();
            })

            Row() {
              Text(this.hasScrolledToBottom ? '已阅读完毕' : '请阅读完整协议内容')
                .fontSize(12)
                .fontColor(this.hasScrolledToBottom ? $r('app.color.primary') : $r('app.color.text_tertiary'))
            }
            .width('100%')
            .justifyContent(FlexAlign.Center)
            .padding({ top: 8, bottom: 8 })
            .backgroundColor($r('app.color.background_tertiary'))
          }
          .width('100%')

          Row({ space: 16 }) {
            Button('不同意')
              .layoutWeight(1)
              .height(44)
              .fontSize(15)
              .fontColor($r('app.color.text_primary'))
              .backgroundColor($r('app.color.background_tertiary'))
              .borderRadius(12)
              .onClick(() => {
                this.handleDisagree();
              })

            Button('同意并继续')
              .layoutWeight(1)
              .height(44)
              .fontSize(15)
              .fontColor($r('app.color.background_primary'))
              .backgroundColor(this.hasScrolledToBottom ? $r('app.color.primary') : $r('app.color.text_tertiary'))
              .borderRadius(12)
              .enabled(this.hasScrolledToBottom)
              .onClick(() => {
                this.handleAgree();
              })
          }
          .width('100%')
          .padding({ left: 20, right: 20, top: 16, bottom: 20 })
        }
        .width('90%')
        .backgroundColor($r('app.color.background_primary'))
        .borderRadius(20)
      }
      .width('100%')
      .height('100%')
      .justifyContent(FlexAlign.Center)
      .backgroundColor('rgba(0, 0, 0, 0.5)')
    }
  }

  /**
   * 检查滚动位置
   * 判断是否已滚动到底部
   */
  private checkScrollPosition(): void {
    try {
      const scrollable = this.scroller.isAtEnd();
      if (scrollable) {
        this.hasScrolledToBottom = true;
      }
    } catch (error) {
      console.warn('PrivacyDialog: Failed to check scroll position:', error);
    }
  }

  private handleDisagree(): void {
    this.context.terminateSelf();
  }

  private async handleAgree(): Promise<void> {
    if (!this.hasScrolledToBottom) {
      return;
    }

    try {
      const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
      await dataPreferences.put(KEY_PRIVACY_AGREED, true);
      await dataPreferences.flush();
      console.info('PrivacyDialog: Privacy agreement saved');
    } catch (error) {
      console.error('PrivacyDialog: Failed to save privacy agreement:', error);
    }

    this.isVisible = false;
    if (this.onAgreed) {
      this.onAgreed();
    }
  }
}

/**
 * 检查用户是否已同意隐私政策
 * 
 * @returns 是否已同意隐私政策
 */
export async function hasAgreedPrivacy(): Promise<boolean> {
  try {
    const dataPreferences = await preferences.getPreferences(getContext(), PREFERENCES_NAME);
    const agreed = await dataPreferences.get(KEY_PRIVACY_AGREED, false);
    return Boolean(agreed);
  } catch (error) {
    console.error('PrivacyDialog: Failed to check privacy agreement:', error);
    return false;
  }
}
