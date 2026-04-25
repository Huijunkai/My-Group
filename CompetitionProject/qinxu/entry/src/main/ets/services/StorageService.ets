/**
 * 存储服务模块
 * 
 * 提供统一的本地存储功能，包括：
 * - 键值对存储
 * - 泛型数据存取
 * - 数据清理
 * 
 * @module StorageService
 */

import { preferences } from '@kit.ArkData';
import { Context } from '@kit.AbilityKit';

/**
 * 存储服务接口
 */
export interface IStorageService {
  /** 初始化存储服务
   * @param context - 应用上下文
   */
  initialize(context: Context): Promise<void>;
  /** 获取数据
   * @param key - 键名
   * @param defaultValue - 默认值
   * @returns 存储的值或默认值
   */
  get<T>(key: string, defaultValue: T): Promise<T>;
  /** 设置数据
   * @param key - 键名
   * @param value - 值
   */
  set<T>(key: string, value: T): Promise<void>;
  /** 删除数据
   * @param key - 键名
   */
  delete(key: string): Promise<void>;
  /** 清空所有数据 */
  clear(): Promise<void>;
}

/**
 * 存储服务类
 * 
 * 提供统一的本地存储接口
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const storage = StorageService.getInstance('my_store');
 * await storage.initialize(context);
 * await storage.set('key', 'value');
 * const value = await storage.get('key', 'default');
 * ```
 */
export class StorageService implements IStorageService {
  /** 单例实例 */
  private static instance: StorageService;
  /** 偏好设置对象 */
  private dataPreferences: preferences.Preferences | null = null;
  /** 应用上下文 */
  private context: Context | null = null;
  /** 存储名称 */
  private storeName: string;

  private constructor(storeName: string = 'app_storage') {
    this.storeName = storeName;
  }

  static getInstance(storeName?: string): StorageService {
    if (!StorageService.instance) {
      StorageService.instance = new StorageService(storeName);
    }
    return StorageService.instance;
  }

  async initialize(context: Context): Promise<void> {
    this.context = context;
    this.dataPreferences = await preferences.getPreferences(context, this.storeName);
  }

  async get<T>(key: string, defaultValue: T): Promise<T> {
    if (!this.dataPreferences) {
      console.warn('StorageService not initialized');
      return defaultValue;
    }
    return await this.dataPreferences.get(key, defaultValue) as T;
  }

  async set<T>(key: string, value: T): Promise<void> {
    if (!this.dataPreferences) {
      console.warn('StorageService not initialized');
      return;
    }
    await this.dataPreferences.put(key, value);
    await this.dataPreferences.flush();
  }

  async delete(key: string): Promise<void> {
    if (!this.dataPreferences) {
      console.warn('StorageService not initialized');
      return;
    }
    await this.dataPreferences.delete(key);
    await this.dataPreferences.flush();
  }

  async clear(): Promise<void> {
    if (!this.dataPreferences) {
      console.warn('StorageService not initialized');
      return;
    }
    await this.dataPreferences.clear();
    await this.dataPreferences.flush();
  }

  async has(key: string): Promise<boolean> {
    if (!this.dataPreferences) {
      return false;
    }
    return await this.dataPreferences.has(key);
  }
}
