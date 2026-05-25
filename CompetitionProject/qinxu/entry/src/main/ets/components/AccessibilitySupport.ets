/**
 * 无障碍支持组件模块
 * 
 * 提供无障碍功能支持，包括：
 * - 屏幕阅读器支持
 * - 触觉反馈
 * - 语音播报功能
 * 
 * @module AccessibilitySupport
 */

import { CourseItem } from '../model/SettingsData';

/**
 * 无障碍支持组件
 * 
 * 为课程信息提供无障碍访问支持，包括屏幕阅读器描述和触觉反馈
 * 
 * @example
 * ```typescript
 * AccessibilitySupport({
 *   course: courseItem,
 *   isCurrent: true
 * })
 * ```
 */
@Component
export struct AccessibilitySupport {
  /** 课程数据 */
  @Prop course: CourseItem;
  /** 是否为当前课程 */
  @Prop isCurrent: boolean = false;
  
  build() {
    // 无障碍支持容器
    Column() {
      // 课程信息读屏支持
      this.buildScreenReaderSupport()
      
      // 触觉反馈区域
      this.buildHapticFeedback()
    }
  }
  
  /**
   * 构建屏幕阅读器支持区域
   * 为屏幕阅读器提供完整的课程信息描述
   */
  @Builder
  buildScreenReaderSupport() {
    Column()
      .accessibilityGroup(true)
      .accessibilityText(this.getAccessibilityDescription())
  }
  
  /**
   * 构建触觉反馈区域
   * 点击时提供触觉反馈效果
   */
  @Builder
  buildHapticFeedback() {
    Column()
      .onClick(() => {
        this.triggerHapticFeedback();
      })
      .hoverEffect(HoverEffect.Auto)
  }
  
  /**
   * 获取无障碍描述文本
   * 组装课程完整信息供屏幕阅读器播报
   * 
   * @returns 课程信息的完整描述文本
   */
  private getAccessibilityDescription(): string {
    const parts: string[] = [];
    
    // 课程基本信息
    parts.push(`课程：${this.course.name}`);
    parts.push(`类型：${this.course.type}`);
    
    // 时间信息
    if (this.course.startTime && this.course.endTime) {
      parts.push(`时间：${this.course.startTime}到${this.course.endTime}`);
    }
    
    // 地点信息
    if (this.course.location) {
      parts.push(`地点：${this.course.location}`);
    }
    
    // 教师信息
    if (this.course.teacher) {
      parts.push(`教师：${this.course.teacher}`);
    }
    
    // 周次信息
    if (this.course.weeks.length > 0) {
      parts.push(`上课周次：第${this.course.weeks.join('周、第')}周`);
    }
    
    // 当前状态
    if (this.isCurrent) {
      parts.push('当前课程');
    }
    
    return parts.join('，');
  }
  
  /**
   * 触发触觉反馈
   * 在实际应用中可集成系统触觉反馈API
   */
  private triggerHapticFeedback(): void {
    console.log('触觉反馈触发');
  }
}

/**
 * 语音播报组件
 * 
 * 提供文本转语音播报功能，支持开始和停止控制
 * 
 * @example
 * ```typescript
 * VoiceAnnouncement({
 *   text: '课程即将开始'
 * })
 * ```
 */
@Component
export struct VoiceAnnouncement {
  /** 待播报的文本内容 */
  @Prop text: string;
  /** 是否正在播报中 */
  @State isSpeaking: boolean = false;
  
  build() {
    Button(this.isSpeaking ? '停止播报' : '语音播报')
      .fontSize(12)
      .fontColor('#4a90d9')
      .backgroundColor('#ffffff')
      .border({ width: 1, color: '#4a90d9' })
      .borderRadius(16)
      .padding({ left: 12, right: 12 })
      .onClick(() => {
        this.toggleAnnouncement();
      })
      .accessibilityDescription('语音播报按钮')
  }
  
  /**
   * 切换播报状态
   * 在开始和停止播报之间切换
   */
  private toggleAnnouncement(): void {
    if (this.isSpeaking) {
      this.stopAnnouncement();
    } else {
      this.startAnnouncement();
    }
  }
  
  /**
   * 开始语音播报
   * 在实际应用中需集成TTS（文本转语音）功能
   */
  private startAnnouncement(): void {
    this.isSpeaking = true;
    console.log('开始语音播报：', this.text);
    
    setTimeout(() => {
      this.isSpeaking = false;
    }, 3000);
  }
  
  /**
   * 停止语音播报
   */
  private stopAnnouncement(): void {
    this.isSpeaking = false;
    console.log('停止语音播报');
  }
}