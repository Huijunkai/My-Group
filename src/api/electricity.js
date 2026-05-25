const axios = require('axios');
const db = require('../db');

const XYYXT_BASE_URL = 'http://124.70.92.199:3000/api/xyyxt';

async function getElectricity(username, roomId, campusId, buildingId) {
  try {
    const url = `${XYYXT_BASE_URL}/electricity`;
    const params = {
      username,
      roomId,
      areaId: campusId,
      buildingId
    };

    console.log(`Get electricity: ${username} - ${roomId}`);
    
    const response = await axios.get(url, {
      params: params,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    const result = response.data;
    
    if (result.success) {
      console.log(`Get electricity success: ${result.data?.balance || 0} yuan`);
      return {
        success: true,
        data: result.data
      };
    } else {
      return {
        success: false,
        message: result.message || 'Get electricity failed'
      };
    }
  } catch (error) {
    console.error('Get electricity failed:', error.message);
    return {
      success: false,
      message: 'Request failed: ' + error.message
    };
  }
}

async function saveElectricityReminderSettings(studentId, settings) {
  try {
    const campusId = settings.campusId || settings.campus || '';
    const buildingId = settings.buildingId || settings.building || '';
    
    const existingSettings = await db.ElectricityReminder.findOne({
      where: { studentId }
    });

    if (existingSettings) {
      existingSettings.enabled = settings.enabled;
      existingSettings.threshold = settings.threshold;
      existingSettings.electricityAccount = settings.electricityAccount;
      if (settings.electricityPassword) {
        existingSettings.electricityPassword = settings.electricityPassword;
      }
      existingSettings.roomId = settings.roomId;
      existingSettings.campusId = campusId;
      existingSettings.buildingId = buildingId;
      existingSettings.updatedAt = new Date();
      
      await existingSettings.save();
      
      console.log(`Update electricity reminder settings: ${studentId}`);
      return {
        success: true,
        data: existingSettings
      };
    } else {
      const newSettings = await db.ElectricityReminder.create({
        studentId,
        enabled: settings.enabled,
        threshold: settings.threshold,
        electricityAccount: settings.electricityAccount,
        electricityPassword: settings.electricityPassword || '',
        roomId: settings.roomId,
        campusId: campusId,
        buildingId: buildingId,
        createdAt: new Date(),
        updatedAt: new Date()
      });
      
      console.log(`Create electricity reminder settings: ${studentId}`);
      return {
        success: true,
        data: newSettings
      };
    }
  } catch (error) {
    console.error('Save electricity reminder settings failed:', error.message);
    return {
      success: false,
      message: 'Save failed: ' + error.message
    };
  }
}

async function getElectricityReminderSettings(studentId) {
  try {
    const settings = await db.ElectricityReminder.findOne({
      where: { studentId }
    });

    if (settings) {
      const safeSettings = {
        id: settings.id,
        studentId: settings.studentId,
        enabled: settings.enabled,
        threshold: settings.threshold,
        electricityAccount: settings.electricityAccount,
        hasPassword: !!settings.electricityPassword,
        roomId: settings.roomId,
        campusId: settings.campusId,
        buildingId: settings.buildingId,
        createdAt: settings.createdAt,
        updatedAt: settings.updatedAt
      };
      return {
        success: true,
        data: safeSettings
      };
    } else {
      return {
        success: true,
        data: {
          enabled: false,
          threshold: 10,
          electricityAccount: '',
          hasPassword: false,
          roomId: '',
          campusId: '',
          buildingId: ''
        }
      };
    }
  } catch (error) {
    console.error('Get electricity reminder settings failed:', error.message);
    return {
      success: false,
      message: 'Get failed: ' + error.message
    };
  }
}

async function getAllElectricityReminderSettings() {
  try {
    const settings = await db.ElectricityReminder.findAll({
      where: { enabled: true }
    });

    return {
      success: true,
      data: settings
    };
  } catch (error) {
    console.error('Get all electricity reminder settings failed:', error.message);
    return {
      success: false,
      message: 'Get failed: ' + error.message
    };
  }
}

module.exports = {
  getElectricity,
  saveElectricityReminderSettings,
  getElectricityReminderSettings,
  getAllElectricityReminderSettings
};
