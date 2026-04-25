/**
 * 一信通API客户端模块
 * 
 * 提供与一信通系统通信的功能，包括：
 * - 用户登录
 * - 一卡通余额查询
 * - 电费余额查询
 * - 楼栋/房间查询
 * - 消费记录查询
 * 
 * @module XyyxtApiClient
 */

import http from '@ohos.net.http';

/**
 * 登录数据接口
 */
interface XyyxtLoginData {
  /** 访问令牌 */
  access_token: string;
  /** 刷新令牌 */
  refresh_token: string;
  /** 过期时间 */
  expires_in: number;
  /** 学校ID */
  schoolId: string;
  /** 用户ID */
  userId: string;
}

/**
 * 登录响应接口
 */
export interface XyyxtLoginResponse {
  /** 是否成功 */
  success: boolean;
  /** 消息 */
  message: string;
  /** 登录数据 */
  data?: XyyxtLoginData;
}

/**
 * 余额数据接口
 */
interface XyyxtBalanceData {
  /** 余额 */
  balance: number;
  /** 账号 */
  account?: string;
}

/**
 * 余额响应接口
 */
export interface XyyxtBalanceResponse {
  /** 是否成功 */
  success: boolean;
  /** 余额数据 */
  data?: XyyxtBalanceData;
}

/**
 * 电费数据接口
 */
interface XyyxtElectricityData {
  /** 余额 */
  balance: number;
  /** 房间名称 */
  roomName?: string;
}

/**
 * 电费响应接口
 */
export interface XyyxtElectricityResponse {
  /** 是否成功 */
  success: boolean;
  /** 电费数据 */
  data?: XyyxtElectricityData;
}

/**
 * 楼栋接口
 */
export interface XyyxtBuilding {
  /** 楼栋ID */
  loudong_id: string;
  /** 楼栋名称 */
  loudong_name: string;
}

/**
 * 楼栋数据接口
 */
interface XyyxtBuildingsData {
  /** 楼栋列表 */
  data?: XyyxtBuilding[];
}

/**
 * 楼栋响应接口
 */
export interface XyyxtBuildingsResponse {
  /** 是否成功 */
  success: boolean;
  /** 楼栋列表 */
  data?: XyyxtBuilding[];
}

/**
 * 房间接口
 */
export interface XyyxtRoom {
  /** 房间ID */
  room_id: string;
  /** 房间名称 */
  room_name: string;
}

/**
 * 房间数据接口
 */
interface XyyxtRoomsData {
  /** 房间列表 */
  data?: XyyxtRoom[];
}

/**
 * 房间响应接口
 */
export interface XyyxtRoomsResponse {
  /** 是否成功 */
  success: boolean;
  /** 房间列表 */
  data?: XyyxtRoom[];
}

/**
 * 消费记录接口
 */
export interface XyyxtConsumption {
  /** 记录ID */
  id?: string;
  /** 消费原因 */
  reason?: string;
  /** 消费时间 */
  time?: string;
  /** 消费金额 */
  amount?: number;
  /** 余额 */
  balance?: number;
  /** 消费地点 */
  location?: string;
  /** 消费类型 */
  type?: string;
  /** 消费日期 */
  date?: string;
  /** 备注 */
  remark?: string;
}

/**
 * 消费记录响应接口
 */
export interface XyyxtConsumptionsResponse {
  /** 是否成功 */
  success: boolean;
  data?: XyyxtConsumption[];
  total?: number;
  pages?: number;
  current?: number;
}

export interface XyyxtElectricityReminderSettings {
  enabled: boolean;
  threshold: number;
  electricityAccount: string;
  electricityPassword: string;
  roomId: string;
  campus: string;
  building: string;
}

export interface XyyxtElectricityReminderResponse {
  success: boolean;
  message?: string;
  data?: XyyxtElectricityReminderSettings;
}

export class XyyxtApiClient {
  private static instance: XyyxtApiClient;
  private baseUrl: string = 'http://124.70.92.199:3001/api/xyyxt';
  private accessToken: string = '';
  private username: string = '';

  static getInstance(): XyyxtApiClient {
    if (!XyyxtApiClient.instance) {
      XyyxtApiClient.instance = new XyyxtApiClient();
    }
    return XyyxtApiClient.instance;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = url;
  }

  getAccessToken(): string {
    return this.accessToken;
  }

  getUsername(): string {
    return this.username;
  }

  async login(username: string, password: string): Promise<XyyxtLoginResponse> {
    try {
      const httpRequest = http.createHttp();
      const response = await httpRequest.request(
        `${this.baseUrl}/login`,
        {
          method: http.RequestMethod.POST,
          header: {
            'Content-Type': 'application/json'
          },
          extraData: JSON.stringify({ username: username, password: password })
        }
      );

      const result: XyyxtLoginResponse = JSON.parse(response.result as string);
      if (result.success && result.data?.access_token) {
        this.accessToken = result.data.access_token;
        this.username = username;
      }
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient login error:', JSON.stringify(error));
      return { success: false, message: '网络请求失败' };
    }
  }

  async getBalance(username: string): Promise<XyyxtBalanceResponse> {
    try {
      const httpRequest = http.createHttp();
      const response = await httpRequest.request(
        `${this.baseUrl}/balance?username=${encodeURIComponent(username)}`,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtBalanceResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getBalance error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async getBuildings(username: string, campusId?: string): Promise<XyyxtBuildingsResponse> {
    try {
      const httpRequest = http.createHttp();
      let url = `${this.baseUrl}/buildings?username=${encodeURIComponent(username)}`;
      if (campusId) {
        url += `&areaId=${campusId}`;
      }
      const response = await httpRequest.request(
        url,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtBuildingsResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getBuildings error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async getRooms(username: string, buildingId: string, campusId?: string): Promise<XyyxtRoomsResponse> {
    try {
      const httpRequest = http.createHttp();
      let url = `${this.baseUrl}/rooms/all?username=${encodeURIComponent(username)}&buildingId=${buildingId}`;
      if (campusId) {
        url += `&areaId=${campusId}`;
      }
      const response = await httpRequest.request(
        url,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtRoomsResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getRooms error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async getElectricity(username: string, roomId: string, campusId?: string, buildingId?: string): Promise<XyyxtElectricityResponse> {
    try {
      const httpRequest = http.createHttp();
      let url = `${this.baseUrl}/electricity?username=${encodeURIComponent(username)}&roomId=${roomId}`;
      if (campusId) {
        url += `&areaId=${campusId}`;
      }
      if (buildingId) {
        url += `&buildingId=${buildingId}`;
      }
      console.info(`[电费请求] URL: ${url}`);
      const response = await httpRequest.request(
        url,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      console.info(`[电费请求] 响应: ${response.result as string}`);
      const result: XyyxtElectricityResponse = JSON.parse(response.result as string);
      console.info(`[电费请求] 解析结果: success=${result.success}, data=${JSON.stringify(result.data)}`);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getElectricity error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async getConsumptions(username: string, startDate?: string, endDate?: string, page: number = 1, size: number = 10): Promise<XyyxtConsumptionsResponse> {
    try {
      const httpRequest = http.createHttp();
      let url = `${this.baseUrl}/consumption?username=${encodeURIComponent(username)}&page=${page}&size=${size}`;
      if (startDate) {
        url += `&startDate=${startDate}`;
      }
      if (endDate) {
        url += `&endDate=${endDate}`;
      }
      const response = await httpRequest.request(
        url,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtConsumptionsResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getConsumptions error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async getRecharges(username: string, startDate?: string, endDate?: string, page: number = 1, size: number = 10): Promise<XyyxtConsumptionsResponse> {
    try {
      const httpRequest = http.createHttp();
      let url = `${this.baseUrl}/recharge?username=${encodeURIComponent(username)}&page=${page}&size=${size}`;
      if (startDate) {
        url += `&startDate=${startDate}`;
      }
      if (endDate) {
        url += `&endDate=${endDate}`;
      }
      const response = await httpRequest.request(
        url,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtConsumptionsResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getRecharges error:', JSON.stringify(error));
      return { success: false };
    }
  }

  async saveElectricityReminderSettings(studentId: string, settings: XyyxtElectricityReminderSettings): Promise<XyyxtElectricityReminderResponse> {
    try {
      const httpRequest = http.createHttp();
      const response = await httpRequest.request(
        `http://124.70.92.199:3001/api/electricity/reminder/settings`,
        {
          method: http.RequestMethod.POST,
          header: {
            'Content-Type': 'application/json'
          },
          extraData: JSON.stringify({ studentId, settings })
        }
      );

      const result: XyyxtElectricityReminderResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient saveElectricityReminderSettings error:', JSON.stringify(error));
      return { success: false, message: '网络请求失败' };
    }
  }

  async getElectricityReminderSettings(studentId: string): Promise<XyyxtElectricityReminderResponse> {
    try {
      const httpRequest = http.createHttp();
      const response = await httpRequest.request(
        `http://124.70.92.199:3001/api/electricity/reminder/settings?studentId=${encodeURIComponent(studentId)}`,
        {
          method: http.RequestMethod.GET,
          header: {
            'Content-Type': 'application/json'
          }
        }
      );

      const result: XyyxtElectricityReminderResponse = JSON.parse(response.result as string);
      httpRequest.destroy();
      return result;
    } catch (error) {
      console.error('XyyxtApiClient getElectricityReminderSettings error:', JSON.stringify(error));
      return { success: false, message: '网络请求失败' };
    }
  }
}
