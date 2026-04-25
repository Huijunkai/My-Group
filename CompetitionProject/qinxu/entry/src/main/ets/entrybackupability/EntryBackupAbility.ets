/**
 * 应用备份扩展能力模块
 * 
 * 提供应用数据备份和恢复功能，包括：
 * - 数据备份处理
 * - 数据恢复处理
 * 
 * @module EntryBackupAbility
 */

import { hilog } from '@kit.PerformanceAnalysisKit';
import { BackupExtensionAbility, BundleVersion } from '@kit.CoreFileKit';

/** 日志域标识 */
const DOMAIN = 0x0000;

/**
 * 应用备份扩展能力类
 * 
 * 继承自BackupExtensionAbility，负责应用数据的备份和恢复
 * 
 * @extends BackupExtensionAbility
 */
export default class EntryBackupAbility extends BackupExtensionAbility {
  /**
   * 备份回调
   * 在系统执行备份操作时调用
   */
  async onBackup() {
    hilog.info(DOMAIN, 'testTag', 'onBackup ok');
    await Promise.resolve();
  }

  /**
   * 恢复回调
   * 在系统执行恢复操作时调用
   * 
   * @param bundleVersion - 应用版本信息
   */
  async onRestore(bundleVersion: BundleVersion) {
    hilog.info(DOMAIN, 'testTag', 'onRestore ok %{public}s', JSON.stringify(bundleVersion));
    await Promise.resolve();
  }
}