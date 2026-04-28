/**
 * 成绩查询页面模块
 * 
 * 提供成绩查询和展示功能，包括：
 * - 成绩列表
 * - 学期分组
 * - 成绩搜索
 * - AI学习助手
 * 
 * @module GradeQueryPage
 */

import { router, promptAction } from '@kit.ArkUI';
import { DataStore, Grade } from '../utils/DataStore';
import { StudyAssistant } from '../components/StudyAssistant';
import { GradeData } from '../utils/AIService';
import { SemesterUtils, SemesterInfo } from '../utils/SemesterUtils';

/**
 * 成绩查询页面组件
 * 
 * 展示学生成绩并提供AI分析功能
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/GradeQueryPage' });
 * ```
 */
@Entry
@Component
struct GradeQueryPage {
  /** 成绩列表 */
  @State grades: Array<GradeItem> = [];
  /** 分组后的成绩 */
  @State groupedGrades: Map<string, Array<GradeItem>> = new Map();
  /** 筛选后的成绩 */
  @State filteredGrades: Map<string, Array<GradeItem>> = new Map();
  /** 学生ID */
  @State studentId: string = '';
  /** 加载状态 */
  @State isLoading: boolean = true;
  /** 展开的学期集合 */
  @State expandedSemesters: Set<string> = new Set();
  /** 搜索文本 */
  @State searchText: string = '';
  /** 动画学期映射 */
  @State animatedSemesters: Map<string, boolean> = new Map();
  /** 动画项映射 */
  @State animatedItems: Map<string, boolean> = new Map();
  /** 显示学习助手 */
  @State showStudyAssistant: boolean = false;
  /** 助手成绩数据 */
  @State gradeDataForAssistant: GradeData[] = [];
  /** 入学年份 */
  @State enrollmentYear: string = '';
  /** 当前学期信息 */
  @State currentSemesterInfo: SemesterInfo | null = null;

  aboutToAppear() {
    this.studentId = AppStorage.Get('studentId') as string || '';
    this.loadGrades();
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

        Text('成绩查询')
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
        TextInput({ placeholder: '搜索课程名称', text: this.searchText })
          .layoutWeight(1)
          .height(48)
          .fontSize(14)
          .backgroundColor($r('app.color.background_tertiary'))
          .borderRadius(12)
          .padding({ left: 16, right: 16 })
          .placeholderColor($r('app.color.text_secondary'))
          .onChange((value: string) => {
            this.searchText = value;
            this.filterGrades();
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 16, bottom: 16 })
      .backgroundColor($r('app.color.background_primary'))
      .margin({ top: 8 })

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
          Column({ space: 12 }) {
            if (this.filteredGrades.size === 0) {
              Column() {
                Image($r('app.media.info_circle'))
                  .width(80)
                  .height(80)
                  .fillColor($r('app.color.text_tertiary'))
                Text(this.searchText ? '未找到匹配的课程' : '暂无成绩数据')
                  .fontSize(16)
                  .fontColor($r('app.color.text_secondary'))
                  .margin({ top: 16 })
              }
              .width('100%')
              .padding(16)
              .justifyContent(FlexAlign.Center)
            } else {
              ForEach(this.getSortedSemesters(this.filteredGrades), (semester: string) => {
                Column({ space: 0 }) {
                  this.buildSemesterHeader(semester)
                  
                  if (this.expandedSemesters.has(semester)) {
                    Column({ space: 12 }) {
                      ForEach(this.filteredGrades.get(semester), (item: GradeItem, index: number) => {
                        this.buildGradeItem(item, index, semester, `${semester}_${index}`)
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
                .opacity(this.animatedSemesters.get(semester) ? 1 : 0)
                .translate({
                  x: this.animatedSemesters.get(semester) ? 0 : 20,
                  y: this.animatedSemesters.get(semester) ? 0 : 20
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

      if (this.showStudyAssistant && this.gradeDataForAssistant.length > 0) {
        Column() {
          Column()
            .width('100%')
            .height('25%')
            .backgroundColor('transparent')
            .onClick(() => {
              this.showStudyAssistant = false;
            })

          Column() {
            StudyAssistant({
              grades: this.gradeDataForAssistant,
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
        Text(`${this.filteredGrades.get(semester)?.length || 0}门`)
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

  private toggleSemester(semester: string): void {
    const isExpanding = !this.expandedSemesters.has(semester);
    
    if (isExpanding) {
      this.expandedSemesters.add(semester);
      setTimeout(() => {
        const items = this.filteredGrades.get(semester) || [];
        items.forEach((_, index) => {
          const key = `${semester}_${index}`;
          setTimeout(() => {
            this.animatedItems.set(key, true);
          }, index * 50);
        });
      }, 50);
    } else {
      const items = this.filteredGrades.get(semester) || [];
      for (let i = items.length - 1; i >= 0; i--) {
        const key = `${semester}_${i}`;
        this.animatedItems.set(key, false);
      }
      setTimeout(() => {
        this.expandedSemesters.delete(semester);
      }, 200);
    }
  }

  @Builder
  buildGradeItem(item: GradeItem, index: number, semester: string, key: string) {
    Column({ space: 12 }) {
      Row() {
        Column({ space: 8 }) {
          Text(item.courseName)
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.text_primary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })

          if (item.courseCode) {
            Text(`课程编码：${item.courseCode}`)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
          }

          Row({ space: 20 }) {
            if (item.courseType) {
              Row({ space: 4 }) {
                Image($r('app.media.clock'))
                  .width(12)
                  .height(12)
                  .fillColor($r('app.color.text_tertiary'))
                Text(`课程类型：${item.courseType}`)
                  .fontSize(12)
                  .fontColor($r('app.color.text_secondary'))
              }
            }
            if (item.credit) {
              Row({ space: 4 }) {
                Image($r('app.media.hand_thumbsup_fill'))
                  .width(12)
                  .height(12)
                  .fillColor($r('app.color.text_tertiary'))
                Text(`学分：${item.credit}`)
                  .fontSize(12)
                  .fontColor($r('app.color.text_secondary'))
              }
            }
          }
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)

        Column({ space: 4 }) {
          Text(item.score)
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.getScoreColor(item.score))
          
          if (item.gradePoint) {
            Row({ space: 4 }) {
              Image($r('app.media.book_open_fill'))
                .width(12)
                .height(12)
                .fillColor($r('app.color.text_tertiary'))
              Text(`绩点：${item.gradePoint}`)
                .fontSize(12)
                .fontColor($r('app.color.text_secondary'))
            }
            .alignItems(VerticalAlign.Center)
          }
        }
        .alignItems(HorizontalAlign.End)
      }
      .width('100%')
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 16, bottom: 16 })
    .backgroundColor($r('app.color.glass_background'))
    .borderRadius(12)
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

  private getScoreColor(score: string): Resource | string {
    if (score === '优' || score === '优秀') {
      return $r('app.color.success');
    }
    if (score === '良' || score === '良好') {
      return $r('app.color.primary');
    }
    if (score === '中' || score === '中等') {
      return '#fa8c16';
    }
    if (score === '及格' || score === '合格') {
      return '#722ed1';
    }
    if (score === '不及格' || score === '不合格') {
      return '#f5222d';
    }
    
    const scoreNum = Number(score);
    if (isNaN(scoreNum)) {
      return $r('app.color.text_secondary');
    }
    if (scoreNum >= 90) {
      return $r('app.color.success');
    }
    if (scoreNum >= 80) {
      return $r('app.color.primary');
    }
    if (scoreNum >= 70) {
      return '#fa8c16';
    }
    if (scoreNum >= 60) {
      return '#722ed1';
    }
    return '#f5222d';
  }

  private async loadGrades(): Promise<void> {
    if (!this.studentId) {
      promptAction.showToast({ message: '请先登录后再查看成绩' });
      this.isLoading = false;
      return;
    }

    this.isLoading = true;
    const dataStore = DataStore.getInstance();
    const rawGrades = await dataStore.loadGrades();
    const studentInfo = await dataStore.loadStudentInfo();
    this.isLoading = false;

    if (studentInfo && studentInfo.enrollmentYear) {
      this.enrollmentYear = studentInfo.enrollmentYear;
    } else {
      this.enrollmentYear = SemesterUtils.inferEnrollmentYearFromGrades(rawGrades) || '';
    }

    this.currentSemesterInfo = SemesterUtils.getCurrentSemester();

    const gradeItems: GradeItem[] = [];
    const assistantGradeData: GradeData[] = [];
    
    for (let i = 0; i < rawGrades.length; i++) {
      const g = rawGrades[i];
      const item: GradeItem = {
        courseCode: g.courseCode,
        courseName: g.courseName,
        score: g.score,
        credit: g.credit,
        gradePoint: g.gradePoint,
        courseType: g.courseType,
        examType: g.examType,
        semester: g.semester
      };
      gradeItems.push(item);
      
      const scoreNum = parseFloat(g.score);
      if (!isNaN(scoreNum)) {
        assistantGradeData.push({
          courseName: g.courseName,
          credit: parseFloat(g.credit) || 0,
          grade: g.score,
          score: scoreNum,
          semester: g.semester || '未知学期'
        });
      }
    }
    this.grades = gradeItems;
    this.gradeDataForAssistant = assistantGradeData;

    this.groupGradesBySemester();
    this.filterGrades();
  }

  private filterGrades(): void {
    this.animatedSemesters = new Map();
    this.animatedItems = new Map();
    
    if (!this.searchText.trim()) {
      this.filteredGrades = new Map(this.groupedGrades);
      this.expandedSemesters.clear();
    } else {
      this.filteredGrades = new Map();
      const searchLower = this.searchText.toLowerCase().trim();
      
      this.groupedGrades.forEach((grades: Array<GradeItem>, semester: string) => {
        const filtered = grades.filter((grade: GradeItem) => {
          return grade.courseName.toLowerCase().includes(searchLower);
        });
        
        if (filtered.length > 0) {
          this.filteredGrades.set(semester, filtered);
          this.expandedSemesters.add(semester);
        }
      });
    }
    
    setTimeout(() => {
      const sortedSemesters = this.getSortedSemesters(this.filteredGrades);
      sortedSemesters.forEach((semester, semesterIndex) => {
        setTimeout(() => {
          this.animatedSemesters.set(semester, true);
          
          const items = this.filteredGrades.get(semester) || [];
          items.forEach((_, itemIndex) => {
            const key = `${semester}_${itemIndex}`;
            setTimeout(() => {
              this.animatedItems.set(key, true);
            }, itemIndex * 50);
          });
        }, semesterIndex * 100);
      });
    }, 100);
  }

  private groupGradesBySemester(): void {
    this.groupedGrades = new Map();
    this.animatedSemesters = new Map();
    this.animatedItems = new Map();
    this.grades.forEach((grade: GradeItem) => {
      const sem = grade.semester || '未知学期';
      if (!this.groupedGrades.has(sem)) {
        this.groupedGrades.set(sem, []);
        this.animatedSemesters.set(sem, false);
      }
      this.groupedGrades.get(sem)?.push(grade);
    });
  }

  private getSortedSemesters(gradesMap: Map<string, Array<GradeItem>> = this.groupedGrades): string[] {
    const semesters = Array.from(gradesMap.keys());
    return SemesterUtils.sort(semesters);
  }
}

interface GradeItem {
  courseCode: string;
  courseName: string;
  score: string;
  credit: string;
  gradePoint: string;
  courseType: string;
  examType: string;
  semester: string;
}

export { GradeQueryPage };
