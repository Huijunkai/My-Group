/**
 * 学分进度页面模块
 * 
 * 提供学分完成情况展示，包括：
 * - 各类学分进度
 * - 进度条可视化
 * - 学分统计
 * 
 * @module CreditProgressPage
 */

import { router, promptAction } from '@kit.ArkUI';
import { DataStore, Progress } from '../utils/DataStore';

/**
 * 学分进度项接口
 */
interface ProgressItem {
  /** 学分类别 */
  category: string;
  /** 要求学分 */
  requiredCredits: string;
  /** 已完成学分 */
  completedCredits: string;
}

/**
 * 学分进度页面组件
 * 
 * 展示各类型学分的完成进度
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/CreditProgressPage' });
 * ```
 */
@Entry
@Component
struct CreditProgressPage {
  /** 学分进度数据 */
  @State progressData: ProgressItem[] = [];
  /** 学生ID */
  @State studentId: string = '';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 动画项状态映射 */
  @State animatedItems: Map<number, boolean> = new Map();

  aboutToAppear() {
    this.studentId = AppStorage.Get('studentId') as string || '';
    this.loadProgress();
  }

  private async loadProgress(): Promise<void> {
    if (!this.studentId) {
      this.isLoading = false;
      promptAction.showToast({ message: '请先登录后再查看学分进度' });
      return;
    }

    const dataStore = DataStore.getInstance();
    const rawProgress = await dataStore.loadProgress();
    this.isLoading = false;

    if (rawProgress.length === 0) {
      promptAction.showToast({ message: '暂无学分进度数据' });
      return;
    }

    const items: ProgressItem[] = [];
    for (let i = 0; i < rawProgress.length; i++) {
      const p = rawProgress[i];
      const item: ProgressItem = {
        category: p.category,
        requiredCredits: p.requiredCredits,
        completedCredits: p.completedCredits
      };
      items.push(item);
    }
    this.progressData = items;

    this.animatedItems.clear();
    for (let i = 0; i < this.progressData.length; i++) {
      this.animatedItems.set(i, false);
    }

    setTimeout(() => {
      for (let i = 0; i < this.progressData.length; i++) {
        setTimeout(() => {
          this.animatedItems.set(i, true);
        }, i * 100);
      }
    }, 100);
  }

  private getGroupedProgressData(): ProgressItem[][] {
    const groups: ProgressItem[][] = [];
    for (let i = 0; i < this.progressData.length; i += 2) {
      groups.push(this.progressData.slice(i, i + 2));
    }
    return groups;
  }

  private getPercent(item: ProgressItem): number {
    const required = parseFloat(item.requiredCredits) || 0;
    const completed = parseFloat(item.completedCredits) || 0;
    return required > 0 ? (completed / required) * 100 : 0;
  }

  build() {
    Column() {
      Column()
        .width('100%')
        .height(44)

      Row() {
        Button({
          type: ButtonType.Circle,
          stateEffect: true
        }) {
          Image($r('app.media.chevron_left'))
            .width(24)
            .height(24)
            .fillColor($r('app.color.text_primary'))
        }
        .width(40)
        .height(40)
        .backgroundColor('transparent')
        .onClick(() => {
          router.back();
        })

        Text('学分进度')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Blank()
          .width(40)
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 8, bottom: 8 })

      if (this.isLoading) {
        Column() {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'))
          Text('加载中...')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 12 })
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else if (this.progressData.length === 0) {
        Column() {
          Image($r('app.media.info_circle'))
            .width(80)
            .height(80)
            .fillColor($r('app.color.text_tertiary'))
          Text('暂无学分进度数据')
            .fontSize(16)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 16 })
          Text('请先登录并同步数据')
            .fontSize(13)
            .fontColor($r('app.color.text_tertiary'))
            .margin({ top: 4 })
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else {
        Scroll() {
          Column({ space: 16 }) {
            ForEach(this.getGroupedProgressData(), (group: ProgressItem[], groupIndex: number) => {
              Row({ space: 16 }) {
                ForEach(group, (item: ProgressItem, itemIndex: number) => {
                  Column() {
                    this.buildYearProgress(item)
                  }
                  .opacity(this.animatedItems.get(groupIndex * 2 + itemIndex) ? 1 : 0)
                  .translate({
                    x: this.animatedItems.get(groupIndex * 2 + itemIndex) ? 0 : 20,
                    y: this.animatedItems.get(groupIndex * 2 + itemIndex) ? 0 : 20
                  })
                  .animation({
                    duration: 500,
                    curve: Curve.EaseOut,
                    delay: 0
                  })
                })
              }
              .width('100%')
            })
          }
          .width('100%')
          .padding(16)
        }
        .layoutWeight(1)
        .scrollBar(BarState.Auto)
        .edgeEffect(EdgeEffect.Spring)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildYearProgress(item: ProgressItem) {
    Column({ space: 12 }) {
      Text(item.category)
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .textAlign(TextAlign.Center)

      Column({ space: 8 }) {
        Row() {
          Text(`${item.completedCredits}/${item.requiredCredits} 学分`)
            .fontSize(13)
            .fontColor($r('app.color.text_secondary'))
          Blank()
          Text(`${this.getPercent(item).toFixed(0)}%`)
            .fontSize(13)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.primary'))
        }
        .width('100%')

        Stack() {
          Row() {
            Blank()
          }
          .width('100%')
          .height(8)
          .backgroundColor($r('app.color.info_light'))
          .borderRadius(4)

          Row() {
            Blank()
          }
          .width(`${this.getPercent(item)}%`)
          .height(8)
          .backgroundColor($r('app.color.info'))
          .borderRadius(4)
          .animation({
            duration: 1000,
            tempo: 1.0,
            curve: Curve.EaseOut
          })
        }
        .width('100%')
      }
      .width('100%')
    }
    .width('50%')
    .padding(16)
    .backgroundColor($r('app.color.background_secondary'))
    .borderRadius(16)
    .alignItems(HorizontalAlign.Center)
  }
}

export { CreditProgressPage };
