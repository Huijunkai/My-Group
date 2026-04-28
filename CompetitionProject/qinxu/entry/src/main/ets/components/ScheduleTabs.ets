/**
 * 课表Tab切换组件模块
 * 
 * 提供课表多视图切换功能，包括：
 * - 周视图
 * - 日视图
 * - 列表视图
 * 
 * @module ScheduleTabs
 */

import { CourseItem } from '../model/SettingsData';

/**
 * 课表Tab切换组件
 * 
 * 实现周视图、日视图、列表视图三种展示方式的切换
 * 
 * @example
 * ```typescript
 * ScheduleTabs({
 *   courses: $courseList,
 *   currentWeek: $currentWeek
 * })
 * ```
 */
@Component
export struct ScheduleTabs {
  /** 当前选中的Tab索引 */
  @State currentTab: number = 0;
  /** 课程数据列表（双向绑定） */
  @Link courses: CourseItem[];
  /** 当前周次（双向绑定） */
  @Link currentWeek: number;
  
  /** Tabs控制器 */
  private tabsController: TabsController = new TabsController();
  
  build() {
    Column() {
      // 自定义Tab栏
      this.buildCustomTabBar()
      
      // Tab内容区域
      Tabs({
        barPosition: BarPosition.Start,
        controller: this.tabsController
      }) {
        // 周视图
        TabContent() {
          this.buildWeekView()
        }
        
        // 日视图
        TabContent() {
          this.buildDayView()
        }
        
        // 列表视图
        TabContent() {
          this.buildListView()
        }
      }
      .barWidth('100%')
      .barHeight(0) // 隐藏默认Tab栏，使用自定义
      .animationDuration(0) // 禁用默认动画，使用自定义
      .onChange((index: number) => {
        this.currentTab = index;
      })
    }
  }
  
  /**
   * 构建自定义Tab栏
   * 包含周视图、日视图、列表三个Tab按钮
   */
  @Builder
  buildCustomTabBar() {
    Row({ space: 0 }) {
      // 周视图Tab
      Column() {
        Text('周视图')
          .fontSize(16)
          .fontWeight(this.currentTab === 0 ? FontWeight.Bold : FontWeight.Normal)
          .fontColor(this.currentTab === 0 ? '#4a90d9' : '#8e8e93')
      }
      .layoutWeight(1)
      .height(48)
      .justifyContent(FlexAlign.Center)
      .onClick(() => {
        this.tabsController.changeIndex(0);
      })
      
      // 日视图Tab
      Column() {
        Text('日视图')
          .fontSize(16)
          .fontWeight(this.currentTab === 1 ? FontWeight.Bold : FontWeight.Normal)
          .fontColor(this.currentTab === 1 ? '#4a90d9' : '#8e8e93')
      }
      .layoutWeight(1)
      .height(48)
      .justifyContent(FlexAlign.Center)
      .onClick(() => {
        this.tabsController.changeIndex(1);
      })
      
      // 列表视图Tab
      Column() {
        Text('列表')
          .fontSize(16)
          .fontWeight(this.currentTab === 2 ? FontWeight.Bold : FontWeight.Normal)
          .fontColor(this.currentTab === 2 ? '#4a90d9' : '#8e8e93')
      }
      .layoutWeight(1)
      .height(48)
      .justifyContent(FlexAlign.Center)
      .onClick(() => {
        this.tabsController.changeIndex(2);
      })
    }
    .width('100%')
    .backgroundColor('#ffffff')
    .border({
      width: { bottom: 1 },
      color: '#e5e5ea'
    })
  }
  
  /**
   * 构建周视图
   * 以表格形式展示一周的课程安排
   */
  @Builder
  buildWeekView() {
    Column() {
      // 周视图头部
      this.buildWeekViewHeader()
      
      // 周视图内容
      Scroll() {
        Column() {
          // 时间轴和课程网格
          this.buildTimeCourseGrid()
        }
      }
      .scrollable(ScrollDirection.Vertical)
      .layoutWeight(1)
    }
  }
  
  /**
   * 构建日视图
   * 以时间轴形式展示当天课程
   */
  @Builder
  buildDayView() {
    Column() {
      Text('日视图内容')
        .fontSize(18)
        .fontColor('#1c1c1e')
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
  
  /**
   * 构建列表视图
   * 以扁平列表形式展示所有课程
   */
  @Builder
  buildListView() {
    Column() {
      Text('列表视图内容')
        .fontSize(18)
        .fontColor('#1c1c1e')
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
  
  /**
   * 构建周视图头部
   * 显示当前周次和周次切换手势区域
   */
  @Builder
  buildWeekViewHeader() {
    Row({ space: 12 }) {
      Text('第' + this.currentWeek + '周')
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1c1c1e')
      
      // 周次切换手势区域
      Row()
        .layoutWeight(1)
        .height(40)
        .gesture(
          GestureGroup(GestureMode.Sequence,
            PanGesture({ distance: 5 })
              .onActionStart(() => {
                // 手势开始
              })
              .onActionUpdate((event: GestureEvent) => {
                // 手势更新
              })
              .onActionEnd((event: GestureEvent) => {
                // 手势结束，切换周次
                if (Math.abs(event.offsetX) > 50) {
                  // 切换周次逻辑
                }
              })
          )
        )
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
  }
  
  /**
   * 构建时间轴课程网格
   * 显示时间轴和课程卡片网格
   */
  @Builder
  buildTimeCourseGrid() {
    Text('时间轴课程网格')
      .fontSize(16)
      .fontColor('#8e8e93')
  }
}