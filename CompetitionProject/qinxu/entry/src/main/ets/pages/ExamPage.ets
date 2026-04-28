/**
 * 考试安排页面模块
 * 
 * 提供考试信息展示功能，包括：
 * - 考试列表
 * - 考试状态
 * - 考试详情
 * 
 * @module ExamPage
 */

import { router } from '@kit.ArkUI';
import { DataStore, Exam } from '../utils/DataStore';

/**
 * 考试项接口
 */
interface ExamItem {
  /** 课程名称 */
  courseName: string;
  /** 考试时间 */
  examTime: string;
  /** 考试地点 */
  location: string;
  /** 座位号 */
  seatNumber: string;
  /** 考试类型 */
  examType: string;
  /** 考试状态 */
  status: string;
}

/**
 * 考试安排页面组件
 * 
 * 展示学生的考试安排信息
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/ExamPage' });
 * ```
 */
@Entry
@Component
struct ExamPage {
  /** 考试列表 */
  @State exams: ExamItem[] = [];
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 学生ID */
  @State studentId: string = '';
  /** 动画考试映射 */
  @State animatedExams: Map<number, boolean> = new Map();

  aboutToAppear() {
    this.studentId = AppStorage.Get('studentId') as string || '';
    this.loadExams();
  }

  private async loadExams(): Promise<void> {
    if (!this.studentId) {
      this.isLoading = false;
      return;
    }

    const dataStore = DataStore.getInstance();
    const rawExams = await dataStore.loadExams();
    this.isLoading = false;

    const examItems: ExamItem[] = [];
    for (let i = 0; i < rawExams.length; i++) {
      const e = rawExams[i];
      const examTime = e.examTime ? e.examTime : '';
      const status = this.calculateExamStatus(examTime);
      const item: ExamItem = {
        courseName: e.courseName,
        examTime: examTime,
        location: e.location,
        seatNumber: e.seatNumber,
        examType: e.examType,
        status: status
      };
      examItems.push(item);
    }
    this.exams = examItems;

    this.animatedExams.clear();
    for (let i = 0; i < this.exams.length; i++) {
      this.animatedExams.set(i, false);
    }

    setTimeout(() => {
      for (let i = 0; i < this.exams.length; i++) {
        setTimeout(() => {
          this.animatedExams.set(i, true);
        }, i * 100);
      }
    }, 100);
  }

  private calculateExamStatus(examTime: string): string {
    if (!examTime) {
      return '未开始';
    }

    const timeMatch = examTime.match(/(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})~(\d{2}:\d{2})/);
    if (!timeMatch) {
      return '未开始';
    }

    const date = timeMatch[1];
    const startTime = timeMatch[2];
    const endTime = timeMatch[3];
    const startDateTime = new Date(`${date} ${startTime}`);
    const endDateTime = new Date(`${date} ${endTime}`);
    const now = new Date();

    if (now < startDateTime) {
      return '未开始';
    } else if (now >= startDateTime && now <= endDateTime) {
      return '进行中';
    } else {
      return '已结束';
    }
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

        Text('考试安排')
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
      .backgroundColor($r('app.color.background_primary'))

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
      } else if (this.exams.length === 0) {
        Column() {
          Image($r('app.media.info_circle'))
            .width(80)
            .height(80)
            .fillColor($r('app.color.text_tertiary'))
          Text('暂无考试安排')
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
          Column({ space: 12 }) {
            ForEach(this.exams, (exam: ExamItem, index: number) => {
              Column() {
                this.buildExamCard(exam)
              }
              .opacity(this.animatedExams.get(index) ? 1 : 0)
              .translate({
                x: this.animatedExams.get(index) ? 0 : 20,
                y: this.animatedExams.get(index) ? 0 : 20
              })
              .animation({
                duration: 500,
                curve: Curve.EaseOut,
                delay: 0
              })
            })
          }
          .width('100%')
          .padding(16)
        }
        .scrollBar(BarState.Off)
        .edgeEffect(EdgeEffect.Spring)
        .layoutWeight(1)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildExamCard(exam: ExamItem) {
    Column() {
      Row() {
        Column({ space: 8 }) {
          Text(exam.courseName)
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })

          if (exam.status) {
            Text(exam.status)
              .fontSize(12)
              .fontColor(exam.status === '未开始' ? $r('app.color.info') : exam.status === '进行中' ? '#faad14' : '#52c41d')
              .backgroundColor(exam.status === '未开始' ? $r('app.color.info_light') : exam.status === '进行中' ? 'rgba(250,173,20,0.1)' : 'rgba(82,196,26,0.1)')
              .borderRadius(8)
              .padding({ left: 12, right: 12, top: 4, bottom: 4 })
          }
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')

      Divider()
        .strokeWidth(1)
        .color($r('app.color.divider'))
        .margin({ top: 16, bottom: 16 })

      Row() {
        this.buildInfoItem('app.media.calendar', '考试时间', exam.examTime)
        this.buildInfoItem('app.media.clock', '考试类型', exam.examType || '待定')
      }
      .width('100%')

      Row() {
        this.buildInfoItem('app.media.local_fill', '考试地点', exam.location || '待定')
        this.buildInfoItem('app.media.flag', '座位号', exam.seatNumber || '待定')
      }
      .width('100%')
      .margin({ top: 12 })
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildInfoItem(icon: string, label: string, value: string) {
    Row() {
      Image($r(icon))
        .width(16)
        .height(16)
        .fillColor($r('app.color.primary'))

      Column({ space: 4 }) {
        Text(label)
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
        Text(value)
          .fontSize(14)
          .fontColor($r('app.color.text_primary'))
          .fontWeight(FontWeight.Medium)
          .textAlign(TextAlign.Start)
      }
      .alignItems(HorizontalAlign.Start)
      .margin({ left: 12 })
    }
    .layoutWeight(1)
  }
}

export { ExamPage };
