/**
 * 学期设置组件模块
 * 
 * 提供学期相关设置功能，包括：
 * - 学期名称设置
 * - 开学日期设置
 * - 总周数设置
 * - 当前周数计算
 * 
 * @module SemesterSettings
 */

import { promptAction } from '@kit.ArkUI';

/**
 * 学期设置组件
 * 
 * 用于配置学期基本信息，自动计算当前周数
 * 
 * @example
 * ```typescript
 * SemesterSettings({
 *   semesterName: $semesterName,
 *   semesterStartDate: $startDate,
 *   totalWeeks: $totalWeeks
 * })
 * ```
 */
@Component
export struct SemesterSettings {
  @Link semesterName: string;
  @Link semesterStartDate: string;
  @Link totalWeeks: number;
  
  @State showDatePicker: boolean = false;
  @State showWeekPicker: boolean = false;
  @State tempStartDate: string = '';
  @State tempWeeks: number = 20;

  build() {
    Column({ space: 16 }) {
      // 标题
      Text('学期设置')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333')
        .alignSelf(ItemAlign.Start)
        .margin({ left: 16 })

      // 学期名称设置
      Column({ space: 8 }) {
        Row() {
          Text('学期名称')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333')
            .layoutWeight(1)
          
          Text(this.semesterName)
            .fontSize(14)
            .fontColor('#666')
        }
        .width('100%')
        .padding(16)
        .backgroundColor('#ffffff')
        .borderRadius(12)
        .onClick(() => {
          this.editSemesterName();
        })
      }

      // 开学日期设置
      Column({ space: 8 }) {
        Row() {
          Text('开学日期')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333')
            .layoutWeight(1)
          
          Text(this.formatDate(this.semesterStartDate))
            .fontSize(14)
            .fontColor('#666')
        }
        .width('100%')
        .padding(16)
        .backgroundColor('#ffffff')
        .borderRadius(12)
        .onClick(() => {
          this.showDatePicker = true;
        })
      }

      // 总周数设置
      Column({ space: 8 }) {
        Row() {
          Text('总周数')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333')
            .layoutWeight(1)
          
          Text(`${this.totalWeeks}周`)
            .fontSize(14)
            .fontColor('#666')
        }
        .width('100%')
        .padding(16)
        .backgroundColor('#ffffff')
        .borderRadius(12)
        .onClick(() => {
          this.tempWeeks = this.totalWeeks;
          this.showWeekPicker = true;
        })
      }

      // 当前周数信息
      Column({ space: 8 }) {
        Row() {
          Text('当前周数')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#333')
            .layoutWeight(1)
          
          Text(`第${this.calculateCurrentWeek()}周`)
            .fontSize(14)
            .fontColor('#4a90d9')
            .fontWeight(FontWeight.Bold)
        }
        .width('100%')
        .padding(16)
        .backgroundColor('#f8f9fa')
        .borderRadius(12)
      }

      // 日期选择器
      if (this.showDatePicker) {
        this.buildDatePickerDialog()
      }

      // 周数选择器
      if (this.showWeekPicker) {
        this.buildWeekPickerDialog()
      }
    }
    .width('100%')
    .padding({ left: 16, right: 16 })
  }

  @Builder
  buildDatePickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('选择开学日期')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333')

        // 日期输入
        Row({ space: 8 }) {
          TextInput({ placeholder: 'YYYY-MM-DD' })
            .layoutWeight(1)
            .height(40)
            .fontSize(14)
            .border({ width: 1, color: '#ddd' })
            .borderRadius(8)
            .padding(8)
            .onChange((value: string) => {
              this.tempStartDate = value;
            })
          
          Button('今天')
            .fontSize(12)
            .fontColor('#4a90d9')
            .backgroundColor('#ffffff')
            .border({ width: 1, color: '#4a90d9' })
            .borderRadius(8)
            .padding({ left: 8, right: 8 })
            .onClick(() => {
              this.tempStartDate = this.getTodayDate();
            })
        }

        // 快速选择按钮
        Row({ space: 8 }) {
          Button('春季学期')
            .layoutWeight(1)
            .height(36)
            .fontSize(12)
            .fontColor('#4a90d9')
            .backgroundColor('#ffffff')
            .border({ width: 1, color: '#4a90d9' })
            .borderRadius(8)
            .onClick(() => {
              this.tempStartDate = this.getSpringSemesterDate();
            })
          
          Button('秋季学期')
            .layoutWeight(1)
            .height(36)
            .fontSize(12)
            .fontColor('#4a90d9')
            .backgroundColor('#ffffff')
            .border({ width: 1, color: '#4a90d9' })
            .borderRadius(8)
            .onClick(() => {
              this.tempStartDate = this.getFallSemesterDate();
            })
        }

        // 按钮区域
        Row({ space: 12 }) {
          Button('取消')
            .layoutWeight(1)
            .height(44)
            .fontSize(16)
            .fontColor('#666')
            .backgroundColor('#f5f5f5')
            .borderRadius(12)
            .onClick(() => {
              this.showDatePicker = false;
              this.tempStartDate = '';
            })

          Button('确定')
            .layoutWeight(1)
            .height(44)
            .fontSize(16)
            .fontColor('#ffffff')
            .backgroundColor('#4a90d9')
            .borderRadius(12)
            .onClick(() => {
              this.confirmDateSelection();
            })
        }
        .width('100%')
        .margin({ top: 16 })
      }
      .width('80%')
      .padding(24)
      .backgroundColor('#ffffff')
      .borderRadius(20)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center)
    .backgroundColor('rgba(0,0,0,0.5)')
    .onClick(() => {
      this.showDatePicker = false;
      this.tempStartDate = '';
    })
  }

  /**
   * 编辑学期名称
   * 当前版本暂不支持直接编辑
   */
  private editSemesterName(): void {
    promptAction.showToast({ message: '当前版本暂不支持在此处修改学期名称，请在后续版本中使用设置页编辑。' });
  }

  private confirmDateSelection(): void {
    if (!this.validateDate(this.tempStartDate)) {
      promptAction.showToast({ message: '日期格式不正确，请使用YYYY-MM-DD格式' });
      return;
    }

    this.semesterStartDate = this.tempStartDate;
    this.showDatePicker = false;
    this.tempStartDate = '';
    
    promptAction.showToast({ 
      message: `开学日期已设置为：${this.formatDate(this.semesterStartDate)}` 
    });
  }

  private validateDate(dateStr: string): boolean {
    const regex = /^\d{4}-\d{2}-\d{2}$/;
    if (!regex.test(dateStr)) return false;
    
    const date = new Date(dateStr);
    return date instanceof Date && !isNaN(date.getTime());
  }

  private formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return '未设置';
    
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    
    return `${year}年${month}月${day}日`;
  }

  /**
   * 获取今天的日期
   * 
   * @returns 今天的日期字符串（YYYY-MM-DD格式）
   */
  private getTodayDate(): string {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
  }

  /**
   * 获取春季学期默认开学日期
   * 
   * @returns 春季学期开学日期（当年2月20日）
   */
  private getSpringSemesterDate(): string {
  }

  private getFallSemesterDate(): string {
    const currentYear = new Date().getFullYear();
    return `${currentYear}-09-01`; // 秋季学期通常9月初开学
  }

  private calculateCurrentWeek(): number {
    const today = new Date();
    const startDate = new Date(this.semesterStartDate);
    
    if (isNaN(startDate.getTime())) return 1;
    
    // 计算两个日期之间的天数差
    const timeDiff = today.getTime() - startDate.getTime();
    const dayDiff = Math.ceil(timeDiff / (1000 * 3600 * 24));
    
    // 计算周数（从第1周开始）
    const currentWeek = Math.ceil(dayDiff / 7);
    
    return Math.max(1, Math.min(currentWeek, this.totalWeeks));
  }

  @Builder
  buildWeekPickerDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('设置总周数')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333')

        // 周数选择网格
        Grid() {
          ForEach([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], (week: number) => {
            GridItem() {
              Text(`${week}周`)
                .fontSize(14)
                .fontColor(this.tempWeeks === week ? '#ffffff' : '#4a90d9')
                .backgroundColor(this.tempWeeks === week ? '#4a90d9' : '#ffffff')
                .border({ width: 1, color: '#4a90d9' })
                .borderRadius(8)
                .width('100%')
                .height(40)
                .textAlign(TextAlign.Center)
                .onClick(() => {
                  this.tempWeeks = week;
                })
            }
          })
        }
        .columnsTemplate('1fr 1fr 1fr 1fr')
        .rowsGap(8)
        .columnsGap(8)
        .width('100%')

        // 按钮区域
        Row({ space: 12 }) {
          Button('取消')
            .layoutWeight(1)
            .height(44)
            .fontSize(16)
            .fontColor('#666')
            .backgroundColor('#f5f5f5')
            .borderRadius(12)
            .onClick(() => {
              this.showWeekPicker = false;
            })

          Button('确定')
            .layoutWeight(1)
            .height(44)
            .fontSize(16)
            .fontColor('#ffffff')
            .backgroundColor('#4a90d9')
            .borderRadius(12)
            .onClick(() => {
              this.confirmWeekSelection();
            })
        }
        .width('100%')
        .margin({ top: 16 })
      }
      .width('80%')
      .padding(24)
      .backgroundColor('#ffffff')
      .borderRadius(20)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center)
    .backgroundColor('rgba(0,0,0,0.5)')
    .onClick(() => {
      this.showWeekPicker = false;
    })
  }

  /**
   * 确认周数选择
   * 更新总周数设置
   */
  private confirmWeekSelection(): void {
    this.totalWeeks = this.tempWeeks;
    this.showWeekPicker = false;
    
    promptAction.showToast({ 
      message: `总周数已设置为：${this.totalWeeks}周` 
    });
  }
}