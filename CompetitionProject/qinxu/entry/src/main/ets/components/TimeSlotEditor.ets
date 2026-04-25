/**
 * 上课时间段编辑器组件模块
 * 
 * 提供上课时间段管理功能，包括：
 * - 时间段添加
 * - 时间段编辑
 * - 时间段删除
 * - 时间冲突检测
 * 
 * @module TimeSlotEditor
 */

import { promptAction } from '@kit.ArkUI';
import { TimeSlot } from '../model/SettingsData';

/**
 * 上课时间段编辑器组件
 * 
 * 用于管理每日上课时间段设置
 * 
 * @example
 * ```typescript
 * TimeSlotEditor({
 *   timeSlots: $timeSlotList
 * })
 * ```
 */
@Component
export struct TimeSlotEditor {
  /** 时间段列表（双向绑定） */
  @Link timeSlots: TimeSlot[];
  /** 显示添加对话框标志 */
  @State showAddDialog: boolean = false;
  /** 新时间段的开始时间 */
  @State newStartTime: string = '08:00';
  /** 新时间段的结束时间 */
  @State newEndTime: string = '08:45';
  /** 新时间段的节次名称 */
  @State newPeriodName: string = '第1节';

  build() {
    Column({ space: 16 }) {
      // 标题和添加按钮
      Row() {
        Text('上课时间设置')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333')
          .layoutWeight(1)
        
        Button('添加时间段')
          .fontSize(14)
          .fontColor('#4a90d9')
          .backgroundColor('#ffffff')
          .border({ width: 1, color: '#4a90d9' })
          .borderRadius(16)
          .padding({ left: 12, right: 12 })
          .onClick(() => {
            this.showAddDialog = true;
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16 })

      // 时间槽列表
      if (this.timeSlots.length > 0) {
        Column({ space: 8 }) {
          ForEach(this.timeSlots, (slot: TimeSlot, index: number) => {
            this.buildTimeSlotItem(slot, index);
          })
        }
        .width('100%')
        .padding({ left: 16, right: 16 })
      } else {
        Column() {
          Text('暂无上课时间设置')
            .fontSize(14)
            .fontColor('#999')
            .margin({ top: 40 })
        }
        .width('100%')
        .justifyContent(FlexAlign.Center)
      }

      // 添加时间段对话框
      if (this.showAddDialog) {
        this.buildAddTimeSlotDialog()
      }
    }
    .width('100%')
  }

  /**
   * 构建时间段列表项
   * 
   * @param slot - 时间段数据
   * @param index - 时间段索引
   */
  @Builder
  buildTimeSlotItem(slot: TimeSlot, index: number) {
    Row({ space: 12 }) {
      Column({ space: 4 }) {
        Text(slot.period)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor('#333')
        
        Text(slot.time)
          .fontSize(12)
          .fontColor('#666')
      }
      .layoutWeight(1)
      .justifyContent(FlexAlign.Center)

      // 编辑按钮
      Button('编辑')
        .fontSize(12)
        .fontColor('#4a90d9')
        .backgroundColor('#ffffff')
        .border({ width: 1, color: '#4a90d9' })
        .borderRadius(8)
        .padding({ left: 8, right: 8 })
        .onClick(() => {
          this.editTimeSlot(index);
        })

      // 删除按钮
      Button('删除')
        .fontSize(12)
        .fontColor('#ff6b6b')
        .backgroundColor('#ffffff')
        .border({ width: 1, color: '#ff6b6b' })
        .borderRadius(8)
        .padding({ left: 8, right: 8 })
        .onClick(() => {
          this.deleteTimeSlot(index);
        })
    }
    .width('100%')
    .padding(16)
    .backgroundColor('#ffffff')
    .borderRadius(12)
  }

  @Builder
  buildAddTimeSlotDialog() {
    Column() {
      Column({ space: 16 }) {
        Text('添加上课时间段')
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor('#333')
          .margin({ bottom: 8 })

        // 节次名称
        Column({ space: 8 }) {
          Text('节次名称')
            .fontSize(14)
            .fontColor('#333')
            .alignSelf(ItemAlign.Start)
          
          TextInput({ placeholder: '例如：第1节' })
            .width('100%')
            .height(40)
            .fontSize(14)
            .border({ width: 1, color: '#ddd' })
            .borderRadius(8)
            .padding(8)
            .onChange((value: string) => {
              this.newPeriodName = value;
            })
        }

        // 开始时间
        Column({ space: 8 }) {
          Text('开始时间')
            .fontSize(14)
            .fontColor('#333')
            .alignSelf(ItemAlign.Start)
          
          TextInput({ placeholder: '例如：08:00' })
            .width('100%')
            .height(40)
            .fontSize(14)
            .border({ width: 1, color: '#ddd' })
            .borderRadius(8)
            .padding(8)
            .onChange((value: string) => {
              this.newStartTime = value;
            })
        }

        // 结束时间
        Column({ space: 8 }) {
          Text('结束时间')
            .fontSize(14)
            .fontColor('#333')
            .alignSelf(ItemAlign.Start)
          
          TextInput({ placeholder: '例如：08:45' })
            .width('100%')
            .height(40)
            .fontSize(14)
            .border({ width: 1, color: '#ddd' })
            .borderRadius(8)
            .padding(8)
            .onChange((value: string) => {
              this.newEndTime = value;
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
              this.showAddDialog = false;
              this.resetForm();
            })

          Button('确定')
            .layoutWeight(1)
            .height(44)
            .fontSize(16)
            .fontColor('#ffffff')
            .backgroundColor('#4a90d9')
            .borderRadius(12)
            .onClick(() => {
              this.addTimeSlot();
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
      this.showAddDialog = false;
      this.resetForm();
    })
  }

  /**
   * 添加时间段
   * 验证输入并检查时间冲突后添加
   */
  private addTimeSlot(): void {
    if (!this.newPeriodName.trim()) {
      promptAction.showToast({ message: '请输入节次名称' });
      return;
    }

    if (!this.validateTimeFormat(this.newStartTime) || !this.validateTimeFormat(this.newEndTime)) {
      promptAction.showToast({ message: '时间格式不正确，请使用HH:MM格式' });
      return;
    }

    const newSlot = new TimeSlot(
      `${this.newStartTime}-${this.newEndTime}`,
      this.newPeriodName.trim()
    );

    // 检查时间段是否冲突
    if (this.isTimeSlotConflict(newSlot)) {
      promptAction.showToast({ message: '时间段与其他时间段冲突' });
      return;
    }

    this.timeSlots.push(newSlot);
    
    this.timeSlots.sort((a, b) => {
      const aStart = this.parseTime(a.time.split('-')[0]);
      const bStart = this.parseTime(b.time.split('-')[0]);
      return (aStart || 0) - (bStart || 0);
    });

    this.showAddDialog = false;
    this.resetForm();
    promptAction.showToast({ message: '时间段添加成功' });
  }

  /**
   * 编辑时间段
   * 提供修改节次名称和时间的选项
   * 
   * @param index - 要编辑的时间段索引
   */
  private editTimeSlot(index: number): void {
    promptAction.showDialog({
      title: '编辑时间段',
      message: `当前设置：${slot.period} ${slot.time}`,
      buttons: [
        { text: '修改节次名称', color: '#4a90d9' },
        { text: '修改时间', color: '#4a90d9' },
        { text: '取消', color: '#666666' }
      ]
    }).then((result: promptAction.ShowDialogSuccessResponse) => {
      if (result.index === 0) {
        this.editPeriodName(index);
      } else if (result.index === 1) {
        this.editTimeRange(index);
      }
    });
  }

  /**
   * 编辑节次名称
   * 当前版本暂不支持直接编辑
   * 
   * @param index - 时间段索引
   */
  private editPeriodName(index: number): void {
    promptAction.showToast({ message: '当前版本暂不支持在此处修改节次名称，请稍后在设置中调整。' });
  }

  /**
   * 编辑时间范围
   * 当前版本暂不支持直接编辑
   * 
   * @param index - 时间段索引
   */
  private editTimeRange(index: number): void {
    promptAction.showToast({ message: '当前版本暂不支持在此处修改时间段，请删除后重新添加。' });
  }

  /**
   * 删除时间段
   * 弹出确认对话框后执行删除
   * 
   * @param index - 要删除的时间段索引
   */
  private deleteTimeSlot(index: number): void {
    promptAction.showDialog({
      title: '确认删除',
      message: `确定要删除时间段：${this.timeSlots[index].period} ${this.timeSlots[index].time}吗？`,
      buttons: [
        { text: '删除', color: '#ff6b6b' },
        { text: '取消', color: '#666666' }
      ]
    }).then((result: promptAction.ShowDialogSuccessResponse) => {
      if (result.index === 0) {
        this.timeSlots.splice(index, 1);
        promptAction.showToast({ message: '删除成功' });
      }
    });
  }

  /**
   * 验证时间格式
   * 
   * @param timeStr - 时间字符串（HH:MM格式）
   * @returns 是否为有效时间格式
   */
  private validateTimeFormat(timeStr: string): boolean {
    return /^([01]?[0-9]|2[0-3]):[0-5][0-9]$/.test(timeStr);
  }

  /**
   * 检查时间段冲突
   * 判断新时间段是否与现有时间段重叠
   * 
   * @param newSlot - 新时间段
   * @param excludeSlot - 排除的时间段（用于编辑时排除自身）
   * @returns 是否存在冲突
   */
  private isTimeSlotConflict(newSlot: TimeSlot, excludeSlot?: TimeSlot): boolean {
    const newParts = newSlot.time.split('-');
    const newStart = newParts.length > 0 ? newParts[0] : '';
    const newEnd = newParts.length > 1 ? newParts[1] : '';
    const newStartMinutes = this.parseTime(newStart);
    const newEndMinutes = this.parseTime(newEnd);
    
    if (!newStartMinutes || !newEndMinutes) return true;
    if (newEndMinutes <= newStartMinutes) return true;

    for (const slot of this.timeSlots) {
      if (excludeSlot && slot === excludeSlot) continue;
      
      const slotParts = slot.time.split('-');
      const slotStart = slotParts.length > 0 ? slotParts[0] : '';
      const slotEnd = slotParts.length > 1 ? slotParts[1] : '';
      const slotStartMinutes = this.parseTime(slotStart);
      const slotEndMinutes = this.parseTime(slotEnd);
      
      if (slotStartMinutes && slotEndMinutes) {
        if ((newStartMinutes >= slotStartMinutes && newStartMinutes < slotEndMinutes) ||
            (newEndMinutes > slotStartMinutes && newEndMinutes <= slotEndMinutes) ||
            (newStartMinutes <= slotStartMinutes && newEndMinutes >= slotEndMinutes)) {
          return true;
        }
      }
    }
    
    return false;
  }

  /**
   * 解析时间字符串为分钟数
   * 
   * @param timeStr - 时间字符串（HH:MM格式）
   * @returns 分钟数或null（格式无效时）
   */
  private parseTime(timeStr: string): number | null {
    const match = timeStr.match(/^(\d{1,2}):(\d{2})$/);
    if (!match) return null;
    
    const hours = parseInt(match[1]);
    const minutes = parseInt(match[2]);
    
    if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) {
      return null;
    }
    
    return hours * 60 + minutes;
  }

  /**
   * 重置表单
   * 将表单字段恢复为默认值
   */
  private resetForm(): void {
    this.newStartTime = '08:00';
    this.newEndTime = '08:45';
    this.newPeriodName = '第1节';
  }
}