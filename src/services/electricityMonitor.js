const xyyxt = require('../xyyxt');
const pushService = require('./pushService');
const { UserPushToken, ElectricityReminder } = require('../db/models');
const { checkHealth } = require('../db');
const { decrypt, encrypt } = require('../utils/encryption');

class ElectricityMonitor {
  constructor() {
    this.interval = null;
    this.accessTokens = new Map();
    this.notifiedUsers = new Map();
  }

  start() {
    console.log('电费监控服务启动');
    this.interval = setInterval(() => {
      this.checkAllElectricity();
    }, 60 * 60 * 1000);

    this.checkAllElectricity();
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      console.log('电费监控服务停止');
    }
  }

  async checkAllElectricity() {
    try {
      const isHealthy = await checkHealth();
      if (!isHealthy) {
        console.warn('数据库不可用，跳过本次电费检查');
        return;
      }

      console.log('开始检查所有用户电费');
      const settingsList = await ElectricityReminder.findAll({
        where: { enabled: true }
      });
      
      console.log(`找到 ${settingsList.length} 个启用了电费提醒的用户`);

      for (const setting of settingsList) {
        try {
          await this.checkElectricityForUser(setting);
        } catch (userError) {
          console.error(`检查用户 ${setting.studentId} 电费失败:`, userError.message);
        }
      }

      console.log('电费检查完成');
    } catch (error) {
      console.error('电费监控检查失败:', error.message);
      if (error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT') {
        console.warn('网络连接异常，将在下次定时任务重试');
      }
    }
  }

  async loginAndGetToken(username, password) {
    try {
      const decryptedPassword = decrypt(password);
      const result = await xyyxt.login(username, decryptedPassword);
      
      if (result.success && result.data && result.data.access_token) {
        this.accessTokens.set(username, {
          token: result.data.access_token,
          expiresAt: Date.now() + (result.data.expires_in - 300) * 1000
        });
        return result.data.access_token;
      }
      
      console.error(`登录校园一信通失败 (${username}):`, result.message);
      return null;
    } catch (error) {
      console.error(`登录校园一信通异常 (${username}):`, error.message);
      return null;
    }
  }

  async getAccessToken(setting) {
    const electricityAccount = decrypt(setting.electricityAccount);
    const electricityPassword = setting.electricityPassword;
    
    if (!electricityAccount || !electricityPassword) {
      console.log(`用户 ${setting.studentId} 未设置校园一信通账号或密码，跳过检查`);
      return null;
    }

    const cached = this.accessTokens.get(electricityAccount);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.token;
    }

    return await this.loginAndGetToken(electricityAccount, electricityPassword);
  }

  async checkElectricityForUser(setting) {
    try {
      const { studentId, threshold, roomId, campusId, buildingId } = setting;

      const decryptedRoomId = decrypt(roomId);

      if (!decryptedRoomId) {
        console.log(`用户 ${studentId} 未设置房间号，跳过检查`);
        return;
      }

      const accessToken = await this.getAccessToken(setting);
      if (!accessToken) {
        console.log(`用户 ${studentId} 无法获取访问令牌，跳过检查`);
        return;
      }

      const electricityData = await xyyxt.getElectricity(
        accessToken,
        decryptedRoomId,
        decrypt(campusId),
        decrypt(buildingId)
      );

      if (!electricityData) {
        console.error(`获取用户 ${studentId} 电费失败`);
        return;
      }

      const balance = electricityData.balance || electricityData.elecBalance || 0;
      console.log(`用户 ${studentId} 电费余额: ${balance}元, 阈值: ${threshold}元`);

      if (balance < threshold) {
        const notifiedKey = `${studentId}_${Math.floor(balance)}`;
        const lastNotified = this.notifiedUsers.get(notifiedKey);
        const now = Date.now();
        const oneDayMs = 24 * 60 * 60 * 1000;

        if (lastNotified && (now - lastNotified) < oneDayMs) {
          console.log(`用户 ${studentId} 电费不足已通知过（余额${balance}元），24小时内不重复通知`);
          return;
        }

        await this.sendElectricityReminder(studentId, balance, threshold);
        this.notifiedUsers.set(notifiedKey, now);
      } else {
        const keysToRemove = [];
        for (const [key] of this.notifiedUsers) {
          if (key.startsWith(`${studentId}_`)) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach(k => this.notifiedUsers.delete(k));
      }
    } catch (error) {
      console.error(`检查用户 ${setting.studentId} 电费失败:`, error.message);
    }
  }

  async sendElectricityReminder(studentId, balance, threshold) {
    try {
      const userToken = await UserPushToken.findOne({
        where: { studentId, isActive: true }
      });

      if (!userToken) {
        console.log(`用户 ${studentId} 未注册推送Token，跳过推送`);
        return;
      }

      const numericBalance = typeof balance === 'number' ? balance : parseFloat(balance) || 0;
      const numericThreshold = typeof threshold === 'number' ? threshold : parseFloat(threshold) || 0;

      const result = await pushService.sendPushNotification(
        userToken.pushToken,
        '电费余额不足提醒',
        `您的宿舍电费余额为 ${numericBalance.toFixed(2)} 元，已低于设定的 ${numericThreshold} 元阈值，请及时充值！`,
        'electricity_low',
        {
          balance: numericBalance,
          threshold: numericThreshold,
          timestamp: Date.now()
        },
        { visibilityType: 1, badge: { addNum: 1 } }
      );

      if (result.success) {
        console.log(`成功发送电费提醒给用户 ${studentId}`);
      } else {
        console.error(`发送电费提醒失败 [${studentId}]: ${result.message}${result.code ? ' (code=' + result.code + ')' : ''}`);
        console.error(`[电费推送诊断] Token长度=${userToken.pushToken.length} | Token前缀=${userToken.pushToken.substring(0, 15)}...`);
      }
    } catch (error) {
      console.error(`发送电费提醒失败:`, error.message);
    }
  }

  async checkElectricityForStudent(studentId) {
    try {
      const setting = await ElectricityReminder.findOne({
        where: { studentId, enabled: true }
      });

      if (!setting) {
        return { success: false, message: '未开启电费提醒' };
      }

      await this.checkElectricityForUser(setting);
      return { success: true };
    } catch (error) {
      console.error(`手动检查电费失败:`, error.message);
      return { success: false, message: error.message };
    }
  }
}

module.exports = new ElectricityMonitor();
