const electricityApi = require('../api/electricity');
const pushService = require('./pushService');

class ElectricityMonitor {
  constructor() {
    this.interval = null;
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
      console.log('开始检查所有用户电费');
      const settingsResult = await electricityApi.getAllElectricityReminderSettings();
      
      if (!settingsResult.success || !settingsResult.data) {
        console.error('获取电费提醒设置失败:', settingsResult.message);
        return;
      }

      const settingsList = settingsResult.data;
      console.log(`找到 ${settingsList.length} 个启用了电费提醒的用户`);

      for (const setting of settingsList) {
        await this.checkElectricityForUser(setting);
      }

      console.log('电费检查完成');
    } catch (error) {
      console.error('电费监控检查失败:', error.message);
    }
  }

  async checkElectricityForUser(setting) {
    try {
      const { studentId, threshold, roomId, campusId, buildingId, electricityAccount } = setting;

      if (!electricityAccount) {
        console.log(`用户 ${studentId} 未设置电费账号，跳过检查`);
        return;
      }

      if (!roomId || !campusId || !buildingId) {
        console.log(`用户 ${studentId} 电费设置不完整，跳过检查`);
        return;
      }

      const electricityResult = await electricityApi.getElectricity(
        electricityAccount,
        roomId,
        campusId,
        buildingId
      );

      if (!electricityResult.success || !electricityResult.data) {
        console.error(`获取用户 ${studentId} 电费失败:`, electricityResult.message);
        return;
      }

      const balance = electricityResult.data.balance;
      console.log(`用户 ${studentId} 电费余额: ${balance}元, 阈值: ${threshold}元`);

      if (balance < threshold) {
        await this.sendElectricityReminder(studentId, balance, threshold);
      }
    } catch (error) {
      console.error(`检查用户 ${setting.studentId} 电费失败:`, error.message);
    }
  }

  async sendElectricityReminder(studentId, balance, threshold) {
    try {
      const result = await pushService.notifyElectricityLow(studentId, {
        balance: balance,
        threshold: threshold
      });

      if (result.success) {
        console.log(`成功发送电费提醒给用户 ${studentId}`);
      } else {
        console.error(`发送电费提醒失败:`, result.message);
      }
    } catch (error) {
      console.error(`发送电费提醒失败:`, error.message);
    }
  }
}

module.exports = new ElectricityMonitor();
