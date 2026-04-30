/**
 * 隐私政策页面模块
 * 
 * 展示应用隐私政策和软件说明，包括：
 * - 软件功能说明
 * - 使用前提条件
 * - 隐私政策内容
 * 
 * @module PrivacyPolicyPage
 */

import { router } from '@kit.ArkUI';

/**
 * 隐私政策页面组件
 * 
 * 展示应用的隐私政策和相关说明
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/PrivacyPolicyPage' });
 * ```
 */
@Entry
@Component
struct PrivacyPolicyPage {
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
      Text('隐私政策与软件说明')
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

  @Builder
  buildContent() {
    Scroll() {
      Column({ space: 16 }) {
        this.buildSoftwareDescription()
        this.buildDashedDivider()
        this.buildUsagePrerequisites()
        this.buildDashedDivider()
        this.buildPrivacyPolicy()
      }
      .width('100%')
      .padding(16)
    }
    .layoutWeight(1)
  }

  @Builder
  buildSoftwareDescription() {
    Column() {
      Text('一、软件说明')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 16 })

      Text('【核心声明】')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 8 })

      Text('青序 APP（以下简称"本应用"）系由个人开发者独立开发并运营的非官方移动应用程序。本应用与任何学校、教育机构之间不存在任何形式的合作关系、授权关系，亦未签署数据对接协议或达成合作意向。用户通过自主下载、注册、登录及使用本应用的行为，均属于个人自愿行为，与学校要求或委托无关。特此声明，学校对本应用的功能完整性、系统安全性、数据存储合规性、服务稳定性等方面不承担任何法律责任及义务，用户在使用过程中产生的一切风险需自行承担。')
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .lineHeight(24)
        .width('100%')
        .margin({ bottom: 16 })

      Text('【标识使用限制】')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 8 })

      Text('本应用严格遵循标识使用相关规范，在产品全生命周期管理过程中，包含产品设计、开发及运营等各环节，均未使用任何学校的名称、LOGO、徽章等专属标识。在市场推广层面，无论是广告文案、新闻稿件，还是其他宣传资料，均杜绝通过文字表述制造与学校存在关联的误导性信息；在用户界面设计方面，所有界面元素、图标样式均与学校视觉识别系统保持明确区分；功能模块命名亦规避使用易引发联想的词汇。此外，应用开发团队已构建完善的内部审核体系，针对新增功能、宣传物料、运营活动等内容开展多轮严格审核，确保不出现任何暗示与学校存在关联的信息，切实维护用户的知情权，保障市场竞争秩序的规范性与公正性。')
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .lineHeight(24)
        .width('100%')
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildUsagePrerequisites() {
    Column() {
      Text('【使用前提】')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Text('用户点击"登录/使用"按钮即表示：')
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .lineHeight(24)
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 用户已详细阅读、充分理解并自愿接受本隐私政策及软件说明的全部条款，包括但不限于个人信息收集、使用、存储、共享等相关规定。')
        this.buildPolicyItem('2. 用户同意授权本应用在合理必要范围内，收集设备信息（如 IMEI 号、设备型号、操作系统版本）及网络信息（IP 地址），以确保服务的稳定性与安全性。')
        this.buildPolicyItem('3. 用户知悉并确认，本应用将依据功能需求，收集地理位置、通知权限数据。所有收集的数据仅用于实现对应服务功能，绝不作超范围使用。')
        this.buildPolicyItem('4. 用户理解本隐私政策可能因法律法规变更或业务发展需要进行修订，并同意定期查阅更新内容。继续使用本应用即视为用户接受修订后的条款。')
        this.buildPolicyItem('5. 用户明确知晓，本 APP 为个人独立开发的非官方产品。')
        this.buildPolicyItem('6. 用户同意本隐私政策中关于个人信息收集、存储及使用的所有条款。')
      }
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
      Text('二、隐私政策')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 16 })

      this.buildInfoCollection()
      this.buildDivider()
      this.buildInfoStorage()
      this.buildDivider()
      this.buildUserRights()
      this.buildDivider()
      this.buildDisclaimer()
      this.buildDivider()
      this.buildOtherTerms()
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildInfoCollection() {
    Column() {
      Text('（一）信息收集与使用')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 信息收集范围：为保障本应用系统的安全稳定运行，依据相关法律法规要求，本平台将收集用户主动提交的个人信息。具体涵盖您的用户凭据（学号，密码），账户注册资料（如学生真实姓名，性别）、学习相关信息（个人成绩、培养计划、学分获取进度、课程信息、入学时间、专业和学院名称）、设备识别标识等数据。其中部分信息属于敏感个人信息范畴，涉及用户个人隐私，我们请求的信息将保留在您的设备上，不会以任何云端数据库存储收集。')
        this.buildPolicyItem('2. 信息收集准则：本平台严格恪守《中华人民共和国个人信息保护法》《网络数据安全管理条例》等国家法律法规规定，遵循最小必要原则开展信息收集工作。仅采集实现产品服务功能所必需的个人信息，坚决杜绝超范围收集与产品服务无关的用户数据。')
        this.buildPolicyItem('3. 信息使用目的：本平台所收集的个人信息，仅用于优化产品功能与服务体验、强化账户安全防护体系，以及响应处理用户服务需求。未经用户明确授权同意，本平台承诺不会将用户个人信息用于任何形式的商业推广或第三方营销活动。')
        this.buildPolicyItem('4. 使用用户凭据：您的用户凭据（学号，密码），只会向教务系统发起登录请求，出现的保存密码以及记住密码只会保存在您的设备上。')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  buildInfoStorage() {
    Column() {
      Text('（二）信息存储与保护')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 存储方式：用户个人信息将直接存储于您的本地设备中，设备自带的安全防护机制可在一定程度上抵御信息篡改、泄露及非法访问风险。')
        this.buildPolicyItem('2. 存储期限：用户个人信息的存储周期严格遵循数据最小化原则，仅在实现服务目标所需的最短时间内留存。用户完成账号注销操作后，将立即对本地存储的相关信息进行删除，确保数据隐私安全。')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  buildUserRights() {
    Column() {
      Text('（三）用户权利')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 用户享有对其个人信息的查阅、复制、更正、补充及删除权利。用户可通过本 APP 内置反馈渠道，或经由开发者公示的官方联系方式提交相关申请。')
        this.buildPolicyItem('2. 用户有权随时撤回对个人信息收集与使用的授权同意。需特别说明的是，撤回非必要信息授权不会影响核心服务功能的正常使用；若撤回必要信息授权，可能导致部分功能无法正常运行。')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  buildDisclaimer() {
    Column() {
      Text('（四）免责声明')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 本 APP 开发者作为个人信息处理者，严格依据相关法律法规履行数据安全保护义务。')
        this.buildPolicyItem('2. 针对用户使用本 APP 过程中产生的一切争议（涵盖但不限于账号异常、服务质量争议、数据安全纠纷等），均由用户与开发者基于相关法律法规及本隐私政策约定协商解决。')
        this.buildPolicyItem('3. 开发者仅对本 APP 功能设计及运营的合法性负责。用户利用本 APP 实施的任何违法违规行为，其法律后果及责任均由用户自行承担，开发者不对此类行为产生的后果承担法律责任。')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  buildOtherTerms() {
    Column() {
      Text('（五）其他条款')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
        .margin({ bottom: 12 })

      Column({ space: 10 }) {
        this.buildPolicyItem('1. 若对本隐私政策进行修订，我们将通过 APP 弹窗、站内公告等显著方式向用户进行公示。用户继续使用本应用，即表明已充分知悉并同意修订后的隐私政策条款内容。')
        this.buildPolicyItem('2. 如用户有任何咨询、投诉或权利主张需求，可通过以下方式与我们取得联系：yelan12192023@163.com。我们将及时响应并处理用户反馈。')
        this.buildPolicyItem('3. 本隐私政策及软件说明的最终解释权归本 APP 开发主体所有。未尽事宜，将严格依照国家相关法律法规及监管要求执行。')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  buildPolicyItem(content: string) {
    Text(content)
      .fontSize(14)
      .fontColor($r('app.color.text_secondary'))
      .lineHeight(24)
      .width('100%')
  }

  @Builder
  buildDivider() {
    Column()
      .width('100%')
      .height(1)
      .backgroundColor($r('app.color.divider'))
      .margin({ top: 12, bottom: 12 })
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
      })
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
  }
}

export { PrivacyPolicyPage };
