/**
 * 课程卡片组件模块
 * 
 * 提供课程信息展示卡片，包括：
 * - 课程基本信息展示
 * - 课程类型颜色标识
 * - 选中状态样式
 * 
 * @module CourseCard
 */

import { CourseItem } from '../model/SettingsData';

/**
 * 课程卡片组件
 * 
 * 展示单个课程的详细信息，支持选中状态和点击交互
 * 
 * @example
 * ```typescript
 * CourseCard({
 *   course: courseItem,
 *   isSelected: false,
 *   onTap: () => { console.log('点击课程'); }
 * })
 * ```
 */
@Component
export struct CourseCard {
  /** 课程数据 */
  @Prop course: CourseItem;
  /** 是否选中状态 */
  @Prop isSelected: boolean = false;
  /** 点击回调函数 */
  @Prop onTap: () => void = () => {};
  
  /**
   * 获取课程类型对应的颜色
   * 
   * @param type - 课程类型
   * @returns 十六进制颜色值
   */
  private getCourseColor(type: string): string {
    const colorMap: Record<string, string> = {
      '专业课': '#4a90d9',
      '选修课': '#34c759', 
      '实验课': '#ff9500',
      '体育课': '#ff3b30',
      '讲座': '#af52de'
    };
    return colorMap[type] || '#8e8e93';
  }
  
  build() {
    Column({ space: 8 }) {
      // 课程类型标识条
      Row()
        .width('100%')
        .height(4)
        .backgroundColor(this.getCourseColor(this.course.type || '专业课'))
        .borderRadius(2)
      
      // 课程信息
      Column({ space: 6 }) {
        // 课程名称
        Text(this.course.name)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor('#1c1c1e')
          .maxLines(2)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .layoutWeight(1)
        
        // 时间信息
        Row({ space: 8 }) {
          Image($r('app.media.clock'))
            .width(12)
            .height(12)
          Text(`${this.course.startTime}-${this.course.endTime}`)
            .fontSize(12)
            .fontColor('#8e8e93')
        }
        .width('100%')
        
        // 地点信息
        Row({ space: 8 }) {
          Image($r('app.media.local_fill'))
            .width(12)
            .height(12)
          Text(this.course.location)
            .fontSize(12)
            .fontColor('#8e8e93')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .width('100%')
        
        // 教师信息
        Row({ space: 8 }) {
          Image($r('app.media.person'))
            .width(12)
            .height(12)
          Text(this.course.teacher)
            .fontSize(12)
            .fontColor('#8e8e93')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .width('100%')
      }
      .padding({ top: 8, bottom: 12, left: 12, right: 12 })
    }
    .width('100%')
    .backgroundColor(this.isSelected ? '#f2f8ff' : '#ffffff')
    .border({
      width: this.isSelected ? 2 : 1,
      color: this.isSelected ? '#4a90d9' : '#e5e5ea'
    })
    .borderRadius(12)
    .shadow({
      radius: this.isSelected ? 8 : 4,
      color: this.isSelected ? '#4a90d920' : '#00000010',
      offsetX: 0,
      offsetY: this.isSelected ? 2 : 1
    })
    .onClick(() => {
      this.onTap();
    })
    .animation({
      duration: 200,
      curve: Curve.EaseOut
    })
  }
}