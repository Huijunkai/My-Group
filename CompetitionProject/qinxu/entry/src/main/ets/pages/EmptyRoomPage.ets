/**
 * 空教室查询页面模块
 * 
 * 提供空教室查询功能，包括：
 * - 校区选择
 * - 楼栋选择
 * - 周次和节次筛选
 * - 教室课表查看
 * 
 * @module EmptyRoomPage
 */

import { router } from '@kit.ArkUI';
import { ApiClient, EmptyRoom, CampusItem, BuildingItem, EmptyRoomQuery, EmptySlot, RoomSchedule, DaySchedule } from '../utils/ApiClient';
import { DataStore } from '../utils/DataStore';
import { SettingsData } from '../model/SettingsData';

/**
 * 下拉选项数据接口
 */
interface SelectOptionData {
  /** 选项值 */
  value: string;
}

/**
 * 空教室查询页面组件
 * 
 * 查询指定时间段内的空闲教室
 * 
 * @example
 * ```typescript
 * router.pushUrl({ url: 'pages/EmptyRoomPage' });
 * ```
 */
@Entry
@Component
struct EmptyRoomPage {
  /** 查询模式 */
  @State queryMode: number = 0;
  /** 校区列表 */
  @State campuses: CampusItem[] = [];
  /** 楼栋列表 */
  @State buildings: BuildingItem[] = [];
  /** 选中的校区ID */
  @State selectedCampus: string = '';
  /** 选中的校区名称 */
  @State selectedCampusName: string = '';
  /** 选中的楼栋ID */
  @State selectedBuilding: string = '';
  /** 选中的楼栋名称 */
  @State selectedBuildingName: string = '';
  /** 选中的开始周次 */
  @State selectedWeekStart: number = 1;
  /** 选中的结束周次 */
  @State selectedWeekEnd: number = 1;
  /** 选中的开始节次 */
  @State selectedPeriodStart: string = '';
  /** 选中的结束节次 */
  @State selectedPeriodEnd: string = '';
  /** 空教室列表 */
  @State emptyRooms: EmptyRoom[] = [];
  /** 加载状态 */
  @State isLoading: boolean = false;
  /** 查询中状态 */
  @State isQuerying: boolean = false;
  /** 设置数据 */
  @State settings: SettingsData = new SettingsData();
  /** 当前周次 */
  @State currentWeek: number = 1;
  /** 暗色模式标志（全局存储链接） */
  @StorageLink('isDarkMode') isDarkMode: boolean = false;
  /** 总周数 */
  private totalWeeks: number = 20;
  /** 节次列表 */
  private periods: string[] = ['0102', '0304', '0506', '0708', '0910', '1112'];
  /** 周次选项 */
  private weekOptions: SelectOptionData[] = [];
  /** 节次选项 */
  private periodOptions: SelectOptionData[] = [];
  /** 教室输入 */
  @State roomInput: string = '';
  /** 教室课表 */
  @State roomSchedule: RoomSchedule | null = null;
  /** 教室查询中状态 */
  @State isRoomQuerying: boolean = false;

  aboutToAppear() {
    this.loadSettings();
    this.loadCampuses();
    this.initOptions();
  }

  private initOptions(): void {
    this.weekOptions = [];
    for (let i = 1; i <= this.totalWeeks; i++) {
      const option: SelectOptionData = { value: `第${i}周` };
      this.weekOptions.push(option);
    }
    
    this.periodOptions = [];
    const periodLabels = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节', '第11-12节'];
    for (let i = 0; i < this.periods.length; i++) {
      const option: SelectOptionData = { value: periodLabels[i] };
      this.periodOptions.push(option);
    }
  }

  private async loadSettings(): Promise<void> {
    try {
      const dataStore = DataStore.getInstance();
      const loadedSettings: SettingsData | null = await dataStore.loadSettings();
      if (loadedSettings) {
        this.settings = loadedSettings;
        this.totalWeeks = loadedSettings.totalWeeks || 20;
        this.currentWeek = this.getCurrentWeek();
        this.selectedWeekStart = this.currentWeek;
        this.selectedWeekEnd = this.currentWeek;
        this.initOptions();
      }
    } catch (error) {
      console.error('EmptyRoomPage: Failed to load settings:', error);
    }
  }

  private getCurrentWeek(): number {
    if (!this.settings.semesterStartDate) {
      return 1;
    }
    
    const today = new Date();
    const startDate = new Date(this.settings.semesterStartDate);
    
    if (isNaN(startDate.getTime())) {
      return 1;
    }
    
    const todayOnlyDate = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const startOnlyDate = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate());
    
    const timeDiff = todayOnlyDate.getTime() - startOnlyDate.getTime();
    const dayDiff = Math.ceil(timeDiff / (1000 * 3600 * 24));
    
    const currentWeek = Math.max(1, Math.ceil((dayDiff + 1) / 7));
    
    return Math.min(currentWeek, this.totalWeeks);
  }

  private async loadCampuses(): Promise<void> {
    this.isLoading = true;
    try {
      const data = await ApiClient.getCampuses();
      this.campuses = data;
      console.info('EmptyRoomPage: Campuses loaded:', this.campuses.length);
    } catch (error) {
      console.error('EmptyRoomPage: Failed to load campuses:', JSON.stringify(error));
    } finally {
      this.isLoading = false;
    }
  }

  private async loadBuildings(campusCode: string): Promise<void> {
    if (!campusCode) {
      this.buildings = [];
      return;
    }
    this.isLoading = true;
    try {
      const data = await ApiClient.getBuildings(campusCode);
      this.buildings = data;
      console.info('EmptyRoomPage: Buildings loaded:', this.buildings.length);
    } catch (error) {
      console.error('EmptyRoomPage: Failed to load buildings:', JSON.stringify(error));
    } finally {
      this.isLoading = false;
    }
  }

  private async queryEmptyRooms(): Promise<void> {
    if (!this.selectedCampus) {
      return;
    }

    this.isQuerying = true;
    this.emptyRooms = [];
    try {
      const query: EmptyRoomQuery = {
        semester: this.getCurrentSemester(),
        campus: this.selectedCampus,
        building: this.selectedBuilding,
        weekStart: this.selectedWeekStart,
        weekEnd: this.selectedWeekEnd,
        periodStart: this.selectedPeriodStart,
        periodEnd: this.selectedPeriodEnd
      };
      const data = await ApiClient.queryEmptyRooms(query);
      this.emptyRooms = data;
      console.info('EmptyRoomPage: Empty rooms found:', this.emptyRooms.length);
    } catch (error) {
      console.error('EmptyRoomPage: Failed to query empty rooms:', JSON.stringify(error));
    } finally {
      this.isQuerying = false;
    }
  }

  private async queryRoomSchedule(): Promise<void> {
    if (!this.roomInput || !this.roomInput.trim()) {
      return;
    }

    this.isRoomQuerying = true;
    this.roomSchedule = null;
    try {
      const data = await ApiClient.queryRoomSchedule({
        roomName: this.roomInput.trim(),
        semester: this.getCurrentSemester(),
        campus: this.selectedCampus,
        building: this.selectedBuilding,
        weekStart: this.selectedWeekStart,
        weekEnd: this.selectedWeekEnd,
        periodStart: this.selectedPeriodStart,
        periodEnd: this.selectedPeriodEnd
      });
      this.roomSchedule = data;
      console.info('EmptyRoomPage: Room schedule found');
    } catch (error) {
      console.error('EmptyRoomPage: Failed to query room schedule:', JSON.stringify(error));
    } finally {
      this.isRoomQuerying = false;
    }
  }

  private getCurrentSemester(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1;
    if (month >= 2 && month <= 7) {
      return `${year - 1}-${year}-2`;
    } else if (month >= 8) {
      return `${year}-${year + 1}-1`;
    } else {
      return `${year - 1}-${year}-1`;
    }
  }

  private getCampusOptions(): SelectOptionData[] {
    const options: SelectOptionData[] = [];
    for (let i = 0; i < this.campuses.length; i++) {
      const option: SelectOptionData = { value: this.campuses[i].name };
      options.push(option);
    }
    return options;
  }

  private getBuildingOptions(): SelectOptionData[] {
    const options: SelectOptionData[] = [];
    for (let i = 0; i < this.buildings.length; i++) {
      const option: SelectOptionData = { value: this.buildings[i].name };
      options.push(option);
    }
    return options;
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

        Text('空教室查询')
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

      Scroll() {
        Column({ space: 12 }) {
          this.buildModeSwitch()
          if (this.queryMode === 0) {
            this.buildFilterCard()
            this.buildResultCard()
          } else {
            this.buildRoomQueryCard()
            this.buildRoomScheduleCard()
          }
        }
        .width('100%')
        .padding(16)
      }
      .scrollBar(BarState.Off)
      .edgeEffect(EdgeEffect.Spring)
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_tertiary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP, SafeAreaEdge.BOTTOM])
  }

  @Builder
  buildModeSwitch() {
    Row({ space: 0 }) {
      Button('空教室查询')
        .fontSize(14)
        .fontWeight(this.queryMode === 0 ? FontWeight.Medium : FontWeight.Normal)
        .fontColor(this.queryMode === 0 ? $r('app.color.text_primary') : $r('app.color.text_secondary'))
        .backgroundColor(this.queryMode === 0 ? $r('app.color.background_secondary') : 'transparent')
        .borderRadius(8)
        .layoutWeight(1)
        .onClick(() => {
          this.queryMode = 0;
        })

      Button('教室课表查询')
        .fontSize(14)
        .fontWeight(this.queryMode === 1 ? FontWeight.Medium : FontWeight.Normal)
        .fontColor(this.queryMode === 1 ? $r('app.color.text_primary') : $r('app.color.text_secondary'))
        .backgroundColor(this.queryMode === 1 ? $r('app.color.background_secondary') : 'transparent')
        .borderRadius(8)
        .layoutWeight(1)
        .onClick(() => {
          this.queryMode = 1;
        })
    }
    .width('100%')
    .height(40)
    .padding(4)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(12)
  }

  @Builder
  buildFilterCard() {
    Column({ space: 16 }) {
      Text('查询条件')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      Row({ space: 12 }) {
        Text('校区')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.getCampusOptions())
          .selected(this.campuses.findIndex((c: CampusItem) => c.code === this.selectedCampus))
          .value(this.selectedCampusName || '请选择校区')
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedCampus = this.campuses[index].code;
            this.selectedCampusName = this.campuses[index].name;
            this.selectedBuilding = '';
            this.selectedBuildingName = '';
            this.loadBuildings(this.selectedCampus);
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('教学楼')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.getBuildingOptions())
          .selected(this.buildings.findIndex((b: BuildingItem) => b.code === this.selectedBuilding))
          .value(this.selectedBuildingName || '请选择教学楼')
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedBuilding = this.buildings[index].code;
            this.selectedBuildingName = this.buildings[index].name;
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('周次')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.weekOptions)
          .selected(this.selectedWeekStart - 1)
          .value(`第${this.selectedWeekStart}周`)
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedWeekStart = index + 1;
            if (this.selectedWeekEnd < this.selectedWeekStart) {
              this.selectedWeekEnd = this.selectedWeekStart;
            }
          })

        Text('至')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))

        Select(this.weekOptions)
          .selected(this.selectedWeekEnd - 1)
          .value(`第${this.selectedWeekEnd}周`)
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedWeekEnd = index + 1;
            if (this.selectedWeekStart > this.selectedWeekEnd) {
              this.selectedWeekStart = this.selectedWeekEnd;
            }
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('节次')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.periodOptions)
          .selected(this.selectedPeriodStart ? this.periods.indexOf(this.selectedPeriodStart) : -1)
          .value(this.getPeriodLabel(this.selectedPeriodStart))
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedPeriodStart = this.periods[index];
            if (this.selectedPeriodEnd && this.periods.indexOf(this.selectedPeriodEnd) < index) {
              this.selectedPeriodEnd = this.selectedPeriodStart;
            }
          })

        Text('至')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))

        Select(this.periodOptions)
          .selected(this.selectedPeriodEnd ? this.periods.indexOf(this.selectedPeriodEnd) : -1)
          .value(this.getPeriodLabel(this.selectedPeriodEnd))
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedPeriodEnd = this.periods[index];
            if (this.selectedPeriodStart && this.periods.indexOf(this.selectedPeriodStart) > index) {
              this.selectedPeriodStart = this.selectedPeriodEnd;
            }
          })
      }
      .width('100%')

      Button('查询')
        .width('100%')
        .height(44)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.primary'))
        .borderRadius(12)
        .enabled(!this.isQuerying && this.selectedCampus !== '')
        .onClick(() => {
          this.queryEmptyRooms();
        })
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildResultCard() {
    Column({ space: 12 }) {
      Row() {
        Text('查询结果')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        Blank()

        if (this.emptyRooms.length > 0) {
          Text(`共 ${this.emptyRooms.length} 间`)
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
        }
      }
      .width('100%')

      if (this.isQuerying) {
        Column() {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'))
          Text('查询中...')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 12 })
        }
        .width('100%')
        .padding({ top: 24, bottom: 24 })
      } else if (this.emptyRooms.length === 0) {
        Column({ space: 8 }) {
          Image($r('app.media.info_circle'))
            .width(48)
            .height(48)
            .fillColor($r('app.color.text_tertiary'))
          Text('暂无空教室')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .padding({ top: 24, bottom: 24 })
      } else {
        Flex({ wrap: FlexWrap.Wrap }) {
          ForEach(this.emptyRooms, (room: EmptyRoom) => {
            this.buildRoomItem(room)
          })
        }
        .width('100%')
      }
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildRoomQueryCard() {
    Column({ space: 16 }) {
      Text('查询条件')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      Row({ space: 12 }) {
        Text('校区')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.getCampusOptions())
          .selected(this.campuses.findIndex((c: CampusItem) => c.code === this.selectedCampus))
          .value(this.selectedCampusName || '请选择校区')
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedCampus = this.campuses[index].code;
            this.selectedCampusName = this.campuses[index].name;
            this.selectedBuilding = '';
            this.selectedBuildingName = '';
            this.loadBuildings(this.selectedCampus);
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('教学楼')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.getBuildingOptions())
          .selected(this.buildings.findIndex((b: BuildingItem) => b.code === this.selectedBuilding))
          .value(this.selectedBuildingName || '请选择教学楼')
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedBuilding = this.buildings[index].code;
            this.selectedBuildingName = this.buildings[index].name;
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('教室')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        TextInput({ placeholder: '如: A4308 或 101', text: this.roomInput })
          .fontSize(14)
          .fontColor($r('app.color.text_primary'))
          .placeholderFont({ size: 14 })
          .placeholderColor($r('app.color.text_tertiary'))
          .backgroundColor($r('app.color.background_secondary'))
          .borderRadius(8)
          .layoutWeight(1)
          .onChange((value: string) => {
            this.roomInput = value;
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('周次')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.weekOptions)
          .selected(this.selectedWeekStart - 1)
          .value(`第${this.selectedWeekStart}周`)
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedWeekStart = index + 1;
            if (this.selectedWeekEnd < this.selectedWeekStart) {
              this.selectedWeekEnd = this.selectedWeekStart;
            }
          })

        Text('至')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))

        Select(this.weekOptions)
          .selected(this.selectedWeekEnd - 1)
          .value(`第${this.selectedWeekEnd}周`)
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedWeekEnd = index + 1;
            if (this.selectedWeekStart > this.selectedWeekEnd) {
              this.selectedWeekStart = this.selectedWeekEnd;
            }
          })
      }
      .width('100%')

      Row({ space: 12 }) {
        Text('节次')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
          .width(60)

        Select(this.periodOptions)
          .selected(this.selectedPeriodStart ? this.periods.indexOf(this.selectedPeriodStart) : -1)
          .value(this.getPeriodLabel(this.selectedPeriodStart))
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedPeriodStart = this.periods[index];
            if (this.selectedPeriodEnd && this.periods.indexOf(this.selectedPeriodEnd) < index) {
              this.selectedPeriodEnd = this.selectedPeriodStart;
            }
          })

        Text('至')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))

        Select(this.periodOptions)
          .selected(this.selectedPeriodEnd ? this.periods.indexOf(this.selectedPeriodEnd) : -1)
          .value(this.getPeriodLabel(this.selectedPeriodEnd))
          .font({ size: 14 })
          .fontColor($r('app.color.text_primary'))
          .selectedOptionFont({ size: 14 })
          .optionFont({ size: 14 })
          .layoutWeight(1)
          .onSelect((index: number) => {
            this.selectedPeriodEnd = this.periods[index];
            if (this.selectedPeriodStart && this.periods.indexOf(this.selectedPeriodStart) > index) {
              this.selectedPeriodStart = this.selectedPeriodEnd;
            }
          })
      }
      .width('100%')

      Button('查询课表')
        .width('100%')
        .height(44)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.primary'))
        .borderRadius(12)
        .enabled(!this.isRoomQuerying && this.roomInput.trim() !== '' && this.selectedCampus !== '')
        .onClick(() => {
          this.queryRoomSchedule();
        })
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildRoomScheduleCard() {
    Column({ space: 12 }) {
      Text('课表结果')
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      if (this.isRoomQuerying) {
        Column() {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'))
          Text('查询中...')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 12 })
        }
        .width('100%')
        .padding({ top: 24, bottom: 24 })
      } else if (!this.roomSchedule) {
        Column({ space: 8 }) {
          Image($r('app.media.info_circle'))
            .width(48)
            .height(48)
            .fillColor($r('app.color.text_tertiary'))
          Text('请输入教室名称查询')
            .fontSize(14)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .padding({ top: 24, bottom: 24 })
      } else {
        Text(`教室: ${this.roomSchedule.roomName}`)
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .width('100%')

        this.buildScheduleTable()
      }
    }
    .width('100%')
    .padding(16)
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius(16)
  }

  @Builder
  buildScheduleTable() {
    Scroll() {
      Column({ space: 6 }) {
        Row({ space: 2 }) {
          Text('')
            .fontSize(10)
            .fontColor($r('app.color.text_secondary'))
            .width(40)
            .textAlign(TextAlign.Center)

          ForEach(['一', '二', '三', '四', '五', '六', '日'], (day: string) => {
            Text('周' + day)
              .fontSize(9)
              .fontColor($r('app.color.text_secondary'))
              .width(38)
              .textAlign(TextAlign.Center)
          })
        }
        .width('100%')

        ForEach(['0102', '0304', '0506', '0708', '0910', '1112'], (period: string, periodIndex: number) => {
          Row({ space: 2 }) {
            Text(this.getPeriodLabelShort(period))
              .fontSize(9)
              .fontColor($r('app.color.text_secondary'))
              .width(40)
              .textAlign(TextAlign.Center)

            ForEach([1, 2, 3, 4, 5, 6, 7], (day: number) => {
              Text(this.hasCourse(day, period) ? '有课' : '空闲')
                .fontSize(8)
                .fontColor(this.hasCourse(day, period) ? '#e74c3c' : '#27ae60')
                .width(38)
                .height(24)
                .textAlign(TextAlign.Center)
                .backgroundColor(this.hasCourse(day, period) ? '#fde8e8' : '#e8f8f0')
                .borderRadius(4)
            })
          }
          .width('100%')
        })
      }
      .width('100%')
    }
    .scrollable(ScrollDirection.Horizontal)
    .scrollBar(BarState.Off)
    .width('100%')
  }

  private hasCourse(day: number, period: string): boolean {
    if (!this.roomSchedule || !this.roomSchedule.schedule) return false;
    
    const daySchedule = this.roomSchedule.schedule.find((s: DaySchedule) => s.day === day);
    if (!daySchedule) return false;
    
    return daySchedule.periods.includes(period);
  }

  private getPeriodLabelShort(period: string): string {
    const labels: Record<string, string> = {
      '0102': '1-2节',
      '0304': '3-4节',
      '0506': '5-6节',
      '0708': '7-8节',
      '0910': '9-10节',
      '1112': '11-12节'
    };
    return labels[period] || period;
  }

  @Builder
  buildRoomItem(room: EmptyRoom) {
    Column({ space: 8 }) {
      Row() {
        Text(room.roomName)
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .layoutWeight(1)

        if (room.capacity) {
          Text(`${room.capacity}人`)
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
        }
      }
      .width('100%')

      if (room.emptySlots && room.emptySlots.length > 0) {
        this.buildWeekSchedule(room.emptySlots)
      }
    }
    .width('100%')
    .padding(12)
    .backgroundColor($r('app.color.background_secondary'))
    .borderRadius(10)
    .margin({ bottom: 8 })
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  buildWeekSchedule(emptySlots: EmptySlot[]) {
    Column({ space: 6 }) {
      Text('空闲时间')
        .fontSize(11)
        .fontColor($r('app.color.text_tertiary'))

      Column({ space: 4 }) {
        ForEach(emptySlots, (slot: EmptySlot) => {
          Row({ space: 6 }) {
            Text(this.getDayName(slot.day))
              .fontSize(12)
              .fontColor($r('app.color.primary'))
              .fontWeight(FontWeight.Medium)
              .width(36)

            Text(this.formatPeriods(slot.periods))
              .fontSize(11)
              .fontColor($r('app.color.text_secondary'))
              .layoutWeight(1)
          }
          .width('100%')
        })
      }
      .width('100%')
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }

  private formatPeriods(periods: number[]): string {
    if (!periods || periods.length === 0) return '';
    
    const sorted = [...periods].sort((a, b) => a - b);
    const ranges: string[] = [];
    let start = sorted[0];
    let end = sorted[0];

    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === end + 1) {
        end = sorted[i];
      } else {
        if (start === end) {
          ranges.push(`第${start}节`);
        } else {
          ranges.push(`第${start}-${end}节`);
        }
        start = sorted[i];
        end = sorted[i];
      }
    }

    if (start === end) {
      ranges.push(`第${start}节`);
    } else {
      ranges.push(`第${start}-${end}节`);
    }

    return ranges.join('、');
  }

  private getPeriodLabel(period: string): string {
    if (!period) return '不限';
    const labels: Record<string, string> = {
      '0102': '第1-2节',
      '0304': '第3-4节',
      '0506': '第5-6节',
      '0708': '第7-8节',
      '0910': '第9-10节',
      '1112': '第11-12节'
    };
    return labels[period] || '不限';
  }

  private getDayName(day: number): string {
    const days = ['', '周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    return days[day] || '';
  }
}

export { EmptyRoomPage };
