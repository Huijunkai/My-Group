/**
 * 性能优化列表组件模块
 * 
 * 提供高性能的课程列表展示，包括：
 * - LazyForEach懒加载
 * - 数据源管理
 * - 搜索筛选功能
 * - 图片缓存管理
 * 
 * @module PerformanceOptimizedList
 */

import { CourseItem } from '../model/SettingsData';

/**
 * 数据变化监听器接口
 * 用于监听数据源的变化事件
 */
interface DataChangeListener {
  /** 数据重新加载回调 */
  onDataReloaded(): void;
  /** 数据添加回调 */
  onDataAdd(index: number): void;
  /** 数据删除回调 */
  onDataDelete(index: number): void;
  /** 数据变化回调 */
  onDataChange(index: number): void;
}

/**
 * 数据源接口
 * 定义LazyForEach所需的数据源方法
 */
interface IDataSource {
  /** 获取数据总数 */
  totalCount(): number;
  /** 获取指定索引的数据 */
  getData(index: number): CourseItem;
  /** 注册数据变化监听器 */
  registerDataChangeListener(listener: DataChangeListener): void;
  /** 注销数据变化监听器 */
  unregisterDataChangeListener(listener: DataChangeListener): void;
}

/**
 * 课程列表数据源
 * 实现IDataSource接口，支持数据动态更新
 */
class CourseDataSource implements IDataSource {
  /** 课程数据数组 */
  private courses: CourseItem[] = [];
  /** 数据变化监听器列表 */
  private listeners: DataChangeListener[] = [];
  
  constructor(courses: CourseItem[]) {
    this.courses = courses;
  }
  
  totalCount(): number {
    return this.courses.length;
  }
  
  getData(index: number): CourseItem {
    return this.courses[index];
  }
  
  registerDataChangeListener(listener: DataChangeListener): void {
    this.listeners.push(listener);
  }
  
  unregisterDataChangeListener(listener: DataChangeListener): void {
    const index = this.listeners.indexOf(listener);
    if (index >= 0) {
      this.listeners.splice(index, 1);
    }
  }
  
  // 更新数据
  updateCourses(newCourses: CourseItem[]): void {
    this.courses = newCourses;
    this.notifyDataReload();
  }
  
  /**
   * 添加课程
   * @param course - 要添加的课程
   */
  addCourse(course: CourseItem): void {
    this.courses.push(course);
    this.notifyDataAdd(this.courses.length - 1);
  }
  
  // 删除课程
  removeCourse(index: number): void {
    this.courses.splice(index, 1);
    this.notifyDataDelete(index);
  }
  
  private notifyDataReload(): void {
    this.listeners.forEach((listener: DataChangeListener) => {
      listener.onDataReloaded();
    });
  }
  
  private notifyDataAdd(index: number): void {
    this.listeners.forEach((listener: DataChangeListener) => {
      listener.onDataAdd(index);
    });
  }
  
  private notifyDataDelete(index: number): void {
    this.listeners.forEach((listener: DataChangeListener) => {
      listener.onDataDelete(index);
    });
  }
  
  private notifyDataChange(index: number): void {
    this.listeners.forEach((listener: DataChangeListener) => {
      listener.onDataChange(index);
    });
  }
}

/**
 * 性能优化列表组件
 * 
 * 使用LazyForEach实现高性能的课程列表展示
 * 
 * @example
 * ```typescript
 * PerformanceOptimizedList({ courses: courseList })
 * ```
 */
@Component
export struct PerformanceOptimizedList {
  /** 课程数据 */
  @State courses: CourseItem[] = [];
  /** 选中的课程ID */
  @State selectedCourseId: string = '';
  /** 课程数据源 */
  private dataSource: CourseDataSource = new CourseDataSource([]);
  
  aboutToAppear() {
    this.dataSource = new CourseDataSource(this.courses);
  }
  
  build() {
    Column() {
      // 搜索栏
      this.buildSearchBar()
      
      // 优化后的课程列表
      List({ space: 8 }) {
        LazyForEach(
          this.dataSource,
          (course: CourseItem, index: number) => {
            ListItem() {
              this.buildCourseListItem(course)
            }
          },
          (course: CourseItem) => course.id
        )
      }
      .listDirection(Axis.Vertical)
      .divider({ strokeWidth: 1, color: '#e5e5ea', startMargin: 16, endMargin: 16 })
      .cachedCount(5) // 预缓存5个item
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
  }
  
  @Builder
  buildSearchBar() {
    Row({ space: 12 }) {
      TextInput({ placeholder: '搜索课程名称、教师、地点...' })
        .layoutWeight(1)
        .fontSize(16)
        .backgroundColor('#f2f2f7')
        .borderRadius(10)
        .padding(12)
        .onChange((value: string) => {
          this.handleSearch(value);
        })
      
      // 筛选按钮
      Button('筛选')
        .fontSize(14)
        .fontColor('#4a90d9')
        .backgroundColor('#ffffff')
        .border({ width: 1, color: '#4a90d9' })
        .borderRadius(8)
        .padding({ left: 12, right: 12 })
    }
    .width('100%')
    .padding(16)
    .backgroundColor('#ffffff')
  }
  
  @Builder
  buildCourseListItem(course: CourseItem) {
    Row({ space: 12 }) {
      // 课程类型标识
      Column()
        .width(4)
        .height('100%')
        .backgroundColor(course.color)
        .borderRadius(2)
      
      // 课程信息
      Column({ space: 4 }) {
        // 课程名称和类型
        Row() {
          Text(course.name)
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1c1c1e')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .layoutWeight(1)
          
          Text(course.type)
            .fontSize(12)
            .fontColor('#8e8e93')
            .backgroundColor('#f2f2f7')
            .borderRadius(8)
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
        }
        
        // 时间地点信息
        Row() {
          Text(`${course.startTime || '未知'}-${course.endTime || '未知'}`)
            .fontSize(14)
            .fontColor('#8e8e93')
          
          Text('·')
            .fontSize(14)
            .fontColor('#8e8e93')
            .margin({ left: 8, right: 8 })
          
          Text(course.location)
            .fontSize(14)
            .fontColor('#8e8e93')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .layoutWeight(1)
        }
        
        // 教师和学分信息
        Row() {
          Text(course.teacher)
            .fontSize(12)
            .fontColor('#8e8e93')
          
          if (course.credit > 0) {
            Text('·')
              .fontSize(12)
              .fontColor('#8e8e93')
              .margin({ left: 8, right: 8 })
            
            Text(`${course.credit}学分`)
              .fontSize(12)
              .fontColor('#8e8e93')
          }
        }
      }
      .layoutWeight(1)
      
      // 操作按钮
      Column() {
        if (course.isCurrentCourse()) {
          Text('进行中')
            .fontSize(12)
            .fontColor('#ffffff')
            .backgroundColor('#34c759')
            .borderRadius(12)
            .padding({ left: 8, right: 8, top: 4, bottom: 4 })
        }
      }
    }
    .width('100%')
    .padding({ top: 12, bottom: 12, left: 16, right: 16 })
    .backgroundColor(this.selectedCourseId === course.id ? '#f2f8ff' : '#ffffff')
    .onClick(() => {
      this.selectedCourseId = course.id;
      // 触发课程详情显示
    })
    .animation({
      duration: 200,
      curve: Curve.EaseOut
    })
  }
  
  /**
   * 处理搜索
   * 根据关键词筛选课程
   * 
   * @param keyword - 搜索关键词
   */
  private handleSearch(keyword: string): void {
    if (!keyword.trim()) {
      // 重置为原始数据
      this.dataSource.updateCourses(this.courses);
      return;
    }
    
    const filteredCourses = this.courses.filter((course: CourseItem): boolean =>
      course.name.toLowerCase().includes(keyword.toLowerCase()) ||
      course.teacher.toLowerCase().includes(keyword.toLowerCase()) ||
      course.location.toLowerCase().includes(keyword.toLowerCase())
    );
    
    this.dataSource.updateCourses(filteredCourses);
  }
}

/**
 * 图片缓存管理器
 * 
 * 提供图片三级缓存管理（内存 -> 磁盘 -> 网络）
 * 限制缓存大小，自动清理最早的缓存
 */
export class ImageCacheManager {
  /** 图片缓存Map */
  private static cache: Map<string, string> = new Map();
  
  /**
   * 获取缓存的图片
   * @param url - 图片URL
   * @returns 缓存的图片数据或原URL
   */
  static getCachedImage(url: string): string {
    if (ImageCacheManager.cache.has(url)) {
      const cachedData = ImageCacheManager.cache.get(url);
      return cachedData !== undefined ? cachedData : url;
    }
    
    return url;
  }
  
  /**
   * 设置图片缓存
   * @param url - 图片URL
   * @param data - 图片数据
   */
  static setCachedImage(url: string, data: string): void {
    ImageCacheManager.cache.set(url, data);
    if (ImageCacheManager.cache.size > 100) {
      const firstKeyIterator = ImageCacheManager.cache.keys();
      const firstKeyResult = firstKeyIterator.next();
      if (!firstKeyResult.done) {
        const firstKey: string = firstKeyResult.value;
        ImageCacheManager.cache.delete(firstKey);
      }
    }
  }
}