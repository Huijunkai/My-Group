/**
 * 培养计划页面模块
 * 
 * 提供培养计划展示功能，包括：
 * - 学分完成情况
 * - 课程分类展示
 * - 学期分组
 * - AI学习助手
 * 
 * @module TrainingPlanPage
 */

import { router } from '@kit.ArkUI';
import { DataStore, Plan } from '../utils/DataStore';
import { SemesterUtils, SemesterInfo } from '../utils/SemesterUtils';
import { TrainingPlanAssistant } from '../components/TrainingPlanAssistant';
import { TrainingCourseData } from '../utils/AIService';

/**
 * 计划项接口
 */
interface PlanItem {
  /** 学分类别 */
  category: string;
  /** 要求学分 */
  requiredCredits: number;
  /** 已完成学分 */
  completedCredits: number;
  /** 课程列表 */
  courses: string[];
}

/**
 * 计划数据接口
 */
interface PlanData {
  /** 要求学分 */
  requiredCredits: number;
  /** 已完成学分 */
  completedCredits: number;
  /** 课程列表 */
  courses: TrainingCourse[];
}

/**
 * 培养计划课程接口
 */
interface TrainingCourse {
  /** 课程名称 */
  courseName: string;
  /** 课程代码 */
  courseCode: string;
  /** 学分 */
  credit: number;
  /** 总学时 */
  totalHours: number;
  /** 课程类型 */
  courseType: string;
  /** 课程属性 */
  courseAttribute: string;
  /** 考试类型 */
  examType: string;
  /** 学期 */
  semester: string;
}

/**
 * 计划条目接口
 */
interface PlanEntry {
  /** 类别 */
  category: string;
  /** 数据 */
  data: PlanData;
}

/**
 * 培养计划页面组件
 * 
 * 展示学生的培养计划和学分完成情况
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/TrainingPlanPage' });
 * ```
 */
@Entry
@Component
struct TrainingPlanPage {
  /** 专业名称 */
  @State major: string = '';
  /** 计划项列表 */
  @State planItems: Array<PlanItem> = [];
  /** 学生ID */
  @State studentId: string = '';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 总要求学分 */
  @State totalRequiredCredits: number = 0;
  /** 总已完成学分 */
  @State totalCompletedCredits: number = 0;
  /** 分组后的计划 */
  @State groupedPlans: Map<string, Array<TrainingCourse>> = new Map();
  /** 展开的学期集合 */
  @State expandedSemesters: Set<string> = new Set();
  /** 动画计划映射 */
  @State animatedPlans: Map<string, boolean> = new Map();
  /** 动画项映射 */
  @State animatedItems: Map<string, boolean> = new Map();
  /** 显示学习助手 */
  @State showStudyAssistant: boolean = false;
  /** 助手课程数据 */
  @State courseDataForAssistant: TrainingCourseData[] = [];
  /** 入学年份 */
  @State enrollmentYear: string = '';
  /** 当前学期信息 */
  @State currentSemesterInfo: SemesterInfo | null = null;

  aboutToAppear() {
    this.loadPlanData();
  }

  private async loadPlanData(): Promise<void> {
    const storedStudentId = AppStorage.Get('studentId') as string;
    if (!storedStudentId) {
      this.isLoading = false;
      return;
    }
    this.studentId = storedStudentId;

    const dataStore = DataStore.getInstance();
    const rawPlans = await dataStore.loadPlans();
    const courses: TrainingCourse[] = [];
    const assistantCourseData: TrainingCourseData[] = [];

    for (let i = 0; i < rawPlans.length; i++) {
      const plan = rawPlans[i];
      const course: TrainingCourse = {
        courseName: plan.courseName,
        courseCode: plan.courseCode,
        credit: Number(plan.credit || 0),
        totalHours: Number(plan.totalHours || 0),
        courseType: plan.courseType,
        courseAttribute: plan.courseAttribute,
        examType: plan.examType ? plan.examType : '不考试',
        semester: plan.semester
      };
      courses.push(course);

      assistantCourseData.push({
        courseName: plan.courseName,
        courseCode: plan.courseCode,
        credit: Number(plan.credit || 0),
        totalHours: Number(plan.totalHours || 0),
        courseType: plan.courseType || '',
        courseAttribute: plan.courseAttribute || '',
        examType: plan.examType ? plan.examType : '不考试',
        semester: plan.semester || '未知学期'
      });
    }

    this.courseDataForAssistant = assistantCourseData;
    this.groupPlansBySemester(courses);

    const cachedInfo = await dataStore.loadStudentInfo();
    if (cachedInfo) {
      if (cachedInfo.major) {
        this.major = cachedInfo.major;
      }
      if (cachedInfo.enrollmentYear) {
        this.enrollmentYear = cachedInfo.enrollmentYear;
      }
    }

    if (!this.enrollmentYear) {
      this.enrollmentYear = SemesterUtils.inferEnrollmentYearFromGrades(assistantCourseData) || '';
    }

    this.currentSemesterInfo = SemesterUtils.getCurrentSemester();
    this.isLoading = false;
  }

  private groupPlansBySemester(courses: TrainingCourse[]): void {
    this.groupedPlans = new Map();
    this.animatedPlans = new Map();
    for (let i = 0; i < courses.length; i++) {
      const course = courses[i];
      const sem = course.semester || '未知学期';
      if (!this.groupedPlans.has(sem)) {
        this.groupedPlans.set(sem, []);
        this.animatedPlans.set(sem, false);
      }
      this.groupedPlans.get(sem)?.push(course);
    }

    if (this.groupedPlans.size > 0) {
      setTimeout(() => {
        const sortedSemesters = this.getSortedSemesters();
        for (let i = 0; i < sortedSemesters.length; i++) {
          const semester = sortedSemesters[i];
          setTimeout(() => {
            this.animatedPlans.set(semester, true);
          }, i * 100);
        }
      }, 100);
    }
  }

  private getSortedSemesters(): string[] {
    const semesters = Array.from(this.groupedPlans.keys());
    return SemesterUtils.sort(semesters);
  }

  private toggleSemester(semester: string): void {
    const isExpanding = !this.expandedSemesters.has(semester);

    if (isExpanding) {
      this.expandedSemesters.add(semester);
      setTimeout(() => {
        const items = this.groupedPlans.get(semester) || [];
        for (let i = 0; i < items.length; i++) {
          const key = `${semester}_${i}`;
          setTimeout(() => {
            this.animatedItems.set(key, true);
          }, i * 50);
        }
      }, 50);
    } else {
      const items = this.groupedPlans.get(semester) || [];
      for (let i = items.length - 1; i >= 0; i--) {
        const key = `${semester}_${i}`;
        this.animatedItems.set(key, false);
      }
      setTimeout(() => {
        this.expandedSemesters.delete(semester);
      }, 200);
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

        Text('培养计划')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Button({
          type: ButtonType.Circle,
          stateEffect: true
        }) {
          Image($r('app.media.message_on_message'))
            .width(22)
            .height(22)
            .fillColor($r('app.color.primary'))
        }
        .width(40)
        .height(40)
        .backgroundColor('transparent')
        .onClick(() => {
          this.showStudyAssistant = !this.showStudyAssistant;
        })
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 8, bottom: 8 })
      .backgroundColor($r('app.color.background_primary'))

      Row() {
        Text('专业：')
          .fontSize(16)
          .fontColor($r('app.color.text_primary'))
        Text(this.major || '未获取')
          .fontSize(16)
          .fontColor($r('app.color.primary'))
          .fontWeight(FontWeight.Medium)
      }
      .width('100%')
      .padding(16)
      .backgroundColor($r('app.color.background_primary'))
      .margin({ top: 8 })
      .borderRadius(12)
      .shadow({
        radius: 4,
        color: $r('app.color.overlay'),
        offsetX: 0,
        offsetY: 2
      })

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
      } else {
        Scroll() {
          Column({ space: 8 }) {
            if (this.groupedPlans.size === 0) {
              Column() {
                Image($r('app.media.info_circle'))
                  .width(80)
                  .height(80)
                  .fillColor($r('app.color.text_tertiary'))
                Text('暂无培养计划数据')
                  .fontSize(16)
                  .fontColor($r('app.color.text_secondary'))
                  .margin({ top: 16 })
              }
              .width('100%')
              .padding(16)
              .justifyContent(FlexAlign.Center)
            } else {
              ForEach(this.getSortedSemesters(), (semester: string) => {
                Column({ space: 0 }) {
                  this.buildSemesterHeader(semester)

                  if (this.expandedSemesters.has(semester)) {
                    Column({ space: 12 }) {
                      ForEach(this.groupedPlans.get(semester), (course: TrainingCourse, index: number) => {
                        this.buildCourseItem(course, index, semester, `${semester}_${index}`)
                      })
                    }
                    .width('100%')
                    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
                    .backgroundColor($r('app.color.glass_background'))
                    .animation({
                      duration: 300,
                      curve: Curve.FastOutSlowIn
                    })
                  }
                }
                .width('100%')
                .backgroundColor($r('app.color.glass_background'))
                .borderRadius(12)
                .margin({ bottom: 8 })
                .shadow({
                  radius: 4,
                  color: $r('app.color.overlay'),
                  offsetX: 0,
                  offsetY: 2
                })
                .opacity(this.animatedPlans.get(semester) ? 1 : 0)
                .translate({
                  x: this.animatedPlans.get(semester) ? 0 : 20,
                  y: this.animatedPlans.get(semester) ? 0 : 20
                })
                .animation({
                  duration: 500,
                  curve: Curve.EaseOut,
                  delay: 0
                })
              })
            }
          }
          .width('100%')
          .padding(16)
          .alignItems(HorizontalAlign.Start)
          .justifyContent(FlexAlign.Start)
        }
        .layoutWeight(1)
        .scrollBar(BarState.Auto)
        .edgeEffect(EdgeEffect.Spring)
        .align(Alignment.TopStart)
      }

      if (this.showStudyAssistant && this.courseDataForAssistant.length > 0) {
        Column() {
          Column()
            .width('100%')
            .height('25%')
            .backgroundColor('transparent')
            .onClick(() => {
              this.showStudyAssistant = false;
            })

          Column() {
            TrainingPlanAssistant({
              courses: this.courseDataForAssistant,
              major: this.major,
              enrollmentYear: this.enrollmentYear,
              currentSemesterInfo: this.currentSemesterInfo
            })
          }
          .width('100%')
          .height('75%')
          .backgroundColor($r('app.color.background_primary'))
          .borderRadius({ topLeft: 24, topRight: 24 })
          .shadow({
            radius: 16,
            color: $r('app.color.overlay'),
            offsetX: 0,
            offsetY: -4
          })
        }
        .width('100%')
        .height('100%')
        .position({ x: 0, y: 0 })
        .backgroundColor('rgba(0, 0, 0, 0.3)')
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildSemesterHeader(semester: string) {
    Row() {
      Text(SemesterUtils.format(semester))
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
      Blank()
      Row({ space: 8 }) {
        Text(`${this.groupedPlans.get(semester)?.length || 0}门`)
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
        Image($r('app.media.chevron_right'))
          .width(14)
          .height(14)
          .fillColor($r('app.color.text_tertiary'))
          .rotate({ angle: this.expandedSemesters.has(semester) ? 90 : 0 })
          .animation({
            duration: 300,
            curve: Curve.FastOutSlowIn
          })
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 16, bottom: 16 })
    .backgroundColor(this.expandedSemesters.has(semester) ? $r('app.color.info_light') : $r('app.color.background_primary'))
    .borderRadius(12)
    .animation({
      duration: 300,
      curve: Curve.FastOutSlowIn
    })
    .onClick(() => {
      this.toggleSemester(semester);
    })
  }

  @Builder
  buildCourseItem(course: TrainingCourse, index: number, semester: string, key: string) {
    Column({ space: 8 }) {
      Row() {
        Column({ space: 6 }) {
          Row({ space: 8 }) {
            Text(course.courseName)
              .fontSize(15)
              .fontWeight(FontWeight.Medium)
              .fontColor(course.examType === '考试' ? '#f5222d' : $r('app.color.text_primary'))
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .layoutWeight(1)

            if (course.courseAttribute) {
              Row({ space: 4 }) {
                Image($r('app.media.list_checkmask'))
                  .width(14)
                  .height(14)
                  .fillColor(this.getCourseTypeColor(course.courseAttribute))
                Text(this.getCourseTypeText(course.courseAttribute))
                  .fontSize(10)
                  .fontColor(this.getCourseTypeColor(course.courseAttribute))
                  .fontWeight(FontWeight.Medium)
              }
              .padding({ left: 6, right: 6, top: 2, bottom: 2 })
              .backgroundColor(this.getCourseTypeBg(course.courseAttribute))
              .borderRadius(4)
            }
          }

          if (course.courseCode) {
            Text(`课程编码：${course.courseCode}`)
              .fontSize(11)
              .fontColor($r('app.color.text_secondary'))
          }

          Row({ space: 16 }) {
            Row({ space: 4 }) {
              Image($r('app.media.book_open_fill'))
                .width(12)
                .height(12)
                .fillColor($r('app.color.text_tertiary'))
              Text(`学分：${course.credit}`)
                .fontSize(11)
                .fontColor($r('app.color.text_secondary'))
            }
            Row({ space: 4 }) {
              Image($r('app.media.clock'))
                .width(12)
                .height(12)
                .fillColor($r('app.color.text_tertiary'))
              Text(`学时：${course.totalHours}`)
                .fontSize(11)
                .fontColor($r('app.color.text_secondary'))
            }
            Row({ space: 4 }) {
              Image($r('app.media.exclamationmark'))
                .width(12)
                .height(12)
                .fillColor($r('app.color.text_tertiary'))
              Text(`考核：${course.examType}`)
                .fontSize(11)
                .fontColor(course.examType === '考试' ? '#f5222d' : $r('app.color.text_secondary'))
            }
          }
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)
      }
      .width('100%')
    }
    .width('100%')
    .padding({ left: 14, right: 14, top: 12, bottom: 12 })
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(10)
    .shadow({
      radius: 2,
      color: $r('app.color.overlay'),
      offsetX: 0,
      offsetY: 1
    })
    .opacity(this.animatedItems.get(key) ?? false ? 1 : 0)
    .translate({
      x: 0,
      y: this.animatedItems.get(key) ?? false ? 0 : 20
    })
    .animation({
      duration: 250,
      curve: Curve.FastOutSlowIn
    })
  }

  private getCourseTypeText(type: string): string {
    if (!type) return '未知';
    if (type.includes('必修')) {
      return '必修';
    } else if (type.includes('选修')) {
      return '选修';
    }
    return type;
  }

  private getCourseTypeColor(type: string): Resource | string {
    if (!type) return $r('app.color.text_secondary');
    if (type.includes('必修')) {
      return '#1890ff';
    } else if (type.includes('选修')) {
      return '#52c41a';
    }
    return $r('app.color.text_secondary');
  }

  private getCourseTypeBg(type: string): string {
    if (!type) return 'rgba(200, 200, 200, 0.1)';
    if (type.includes('必修')) {
      return 'rgba(24, 144, 255, 0.1)';
    } else if (type.includes('选修')) {
      return 'rgba(82, 196, 26, 0.1)';
    }
    return 'rgba(200, 200, 200, 0.1)';
  }
}

export { TrainingPlanPage };
