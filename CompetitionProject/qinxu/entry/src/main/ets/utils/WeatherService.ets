/**
 * 天气服务模块
 * 
 * 提供天气信息获取功能，包括：
 * - 实时天气获取
 * - 位置定位
 * - 天气缓存
 * - 分钟级降水预报
 * 
 * @module WeatherService
 */

import { http } from '@kit.NetworkKit';
import { geoLocationManager } from '@kit.LocationKit';

/**
 * 天气数据接口
 */
export interface WeatherData {
  /** 温度 */
  temperature: number;
  /** 体感温度 */
  feelsLike: number;
  /** 天气状况 */
  condition: string;
  /** 天气代码 */
  conditionCode: number;
  /** 湿度 */
  humidity: number;
  /** 风速 */
  windSpeed: number;
  /** 风向 */
  windDirection: string;
  /** 能见度 */
  visibility: number;
  /** 更新时间 */
  updateTime: string;
  /** 位置名称 */
  location: string;
  /** 图标代码 */
  iconCode: number;
  /** 分钟级降水数据 */
  minutely: MinutelyData[];
  /** 最低温度 */
  tempMin: number;
  /** 最高温度 */
  tempMax: number;
  /** 降水概率 */
  precipProbability: number;
}

/**
 * 分钟级降水数据接口
 */
export interface MinutelyData {
  /** 时间 */
  time: string;
  /** 降水量 */
  precipitation: number;
  /** 降水类型 */
  type: string;
}

/**
 * 温度范围接口
 */
export interface TemperatureRange {
  /** 最低温度 */
  tempMin: number;
  /** 最高温度 */
  tempMax: number;
}

/**
 * 位置信息接口
 */
export interface LocationInfo {
  /** 位置名称 */
  name: string;
  /** 位置ID */
  id: string;
}

/**
 * 天气服务类
 * 
 * 提供天气信息获取和缓存功能
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const weatherService = WeatherService.getInstance();
 * const weather = await weatherService.getWeather();
 * ```
 */
export class WeatherService {
  /** 单例实例 */
  private static instance: WeatherService;
  /** 缓存的天气数据 */
  private cachedWeather: WeatherData | null = null;
  /** 最后更新时间 */
  private lastUpdateTime: number = 0;
  /** 缓存有效期（30分钟） */
  private static readonly CACHE_DURATION: number = 30 * 60 * 1000;

  static getInstance(): WeatherService {
    if (!WeatherService.instance) {
      WeatherService.instance = new WeatherService();
    }
    return WeatherService.instance;
  }

  async getWeather(): Promise<WeatherData | null> {
    if (this.cachedWeather && Date.now() - this.lastUpdateTime < WeatherService.CACHE_DURATION) {
      return this.cachedWeather;
    }

    let lat = 23.16;
    let lon = 108.1914;
    let locationName = '南宁';
    let locationId = '101300101';

    try {
      const location = await Promise.race([
        geoLocationManager.getCurrentLocation(),
        new Promise<null>((resolve) => setTimeout(() => resolve(null), 5000))
      ]);
      if (location) {
        lat = location.latitude;
        lon = location.longitude;
        console.info(`WeatherService: Got location: ${lat}, ${lon}`);
        const geoInfo = await this.fetchLocationInfo(lat, lon);
        if (geoInfo) {
          locationName = geoInfo.name;
          locationId = geoInfo.id;
        }
      }
    } catch (error) {
      console.warn('WeatherService: Location not available, using default');
    }

    const weatherData = await this.fetchWeatherFromApi(lat, lon, locationName, locationId);
    if (weatherData) {
      this.cachedWeather = weatherData;
      this.lastUpdateTime = Date.now();
    }
    return weatherData;
  }

  private async fetchLocationInfo(lat: number, lon: number): Promise<LocationInfo | null> {
    try {
      const httpRequest = http.createHttp();
      const url = `https://m849vjd7ga.re.qweatherapi.com/v7/geo/lookup?location=${lon},${lat}&key=99749bc4ec99422ea7441072fc05cd29`;
      console.info(`WeatherService: Requesting geo lookup: ${url}`);

      const response = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        header: { 'Content-Type': 'application/json' },
        connectTimeout: 10000,
        readTimeout: 10000
      });

      console.info(`WeatherService: Geo lookup response: ${response.responseCode}`);
      httpRequest.destroy();

      if (response.responseCode === 200) {
        const result: Record<string, Object> = JSON.parse(response.result as string) as Record<string, Object>;
        const code = result['code'] as string;
        
        if (code === '200' && result['location']) {
          const locationList = result['location'] as Object[];
          if (locationList.length > 0) {
            const locationData = locationList[0] as Record<string, string>;
            console.info(`WeatherService: Location data: ${JSON.stringify(locationData)}`);
            
            const adm1 = locationData['adm1'] ?? '';
            const adm2 = locationData['adm2'] ?? '';
            const name = locationData['name'] ?? '';
            const id = locationData['id'] ?? '';
            
            console.info(`WeatherService: adm1=${adm1}, adm2=${adm2}, name=${name}, id=${id}`);
            
            let displayName = '';
            if (name && adm2 && name !== adm2) {
              displayName = `${adm2} ${name}`;
            } else if (name) {
              displayName = name;
            } else if (adm2) {
              displayName = adm2;
            } else if (adm1) {
              displayName = adm1;
            }
            
            const locationInfo: LocationInfo = {
              name: displayName,
              id: id
            };
            return locationInfo;
          }
        }
      }
      return null;
    } catch (error) {
      console.error('WeatherService: Geo lookup request failed:', JSON.stringify(error));
      return null;
    }
  }

  private async fetchWeatherFromApi(lat: number, lon: number, locationName: string, locationId: string): Promise<WeatherData | null> {
    try {
      const results = await Promise.all([
        this.fetchCurrentWeather(lat, lon, locationName),
        this.fetchMinutelyWeather(lat, lon),
        this.fetchTemperatureRange(lat, lon)
      ]);
      const nowData = results[0];
      const minutelyData = results[1];
      const tempRange = results[2];

      if (nowData) {
        nowData.minutely = minutelyData;
        if (tempRange) {
          nowData.tempMin = tempRange.tempMin;
          nowData.tempMax = tempRange.tempMax;
        }
        const precipProb = this.calculatePrecipProbability(minutelyData);
        nowData.precipProbability = precipProb;
      }
      return nowData;
    } catch (error) {
      console.error('WeatherService: API request failed:', JSON.stringify(error));
      return this.getDefaultWeatherData();
    }
  }

  private async fetchCurrentWeather(lat: number, lon: number, locationName: string): Promise<WeatherData | null> {
    try {
      const httpRequest = http.createHttp();
      const url = `https://m849vjd7ga.re.qweatherapi.com/v7/weather/now?location=${lon},${lat}&key=99749bc4ec99422ea7441072fc05cd29`;
      console.info(`WeatherService: Requesting current weather: ${url}`);

      const response = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        header: { 'Content-Type': 'application/json' },
        connectTimeout: 10000,
        readTimeout: 10000
      });

      console.info(`WeatherService: Current weather response: ${response.responseCode}`);
      httpRequest.destroy();

      if (response.responseCode === 200) {
        const result: Record<string, Object> = JSON.parse(response.result as string) as Record<string, Object>;
        const code = result['code'] as string;
        
        if (code === '200' && result['now']) {
          const now = result['now'] as Record<string, string>;
          return this.parseWeatherData(now, locationName);
        }
      }
      return this.getDefaultWeatherData();
    } catch (error) {
      console.error('WeatherService: Current weather request failed:', JSON.stringify(error));
      return this.getDefaultWeatherData();
    }
  }

  private async fetchMinutelyWeather(lat: number, lon: number): Promise<MinutelyData[]> {
    try {
      const httpRequest = http.createHttp();
      const url = `https://m849vjd7ga.re.qweatherapi.com/v7/minutely/5m?location=${lon},${lat}&key=99749bc4ec99422ea7441072fc05cd29`;
      console.info(`WeatherService: Requesting minutely weather: ${url}`);

      const response = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        header: { 'Content-Type': 'application/json' },
        connectTimeout: 10000,
        readTimeout: 10000
      });

      console.info(`WeatherService: Minutely weather response: ${response.responseCode}`);
      httpRequest.destroy();

      if (response.responseCode === 200) {
        const result: Record<string, Object> = JSON.parse(response.result as string) as Record<string, Object>;
        const code = result['code'] as string;
        
        if (code === '200' && result['minutely']) {
          const minutelyList = result['minutely'] as Object[];
          return this.parseMinutelyData(minutelyList);
        }
      }
      return [];
    } catch (error) {
      console.error('WeatherService: Minutely weather request failed:', JSON.stringify(error));
      return [];
    }
  }

  private calculatePrecipProbability(minutely: MinutelyData[]): number {
    if (!minutely || minutely.length === 0) {
      return 0;
    }
    const rainMinutes = minutely.filter(m => m.precipitation > 0).length;
    return Math.round((rainMinutes / minutely.length) * 100);
  }

  private async fetchTemperatureRange(lat: number, lon: number): Promise<TemperatureRange | null> {
    try {
      const httpRequest = http.createHttp();
      const url = `https://m849vjd7ga.re.qweatherapi.com/v7/weather/24h?location=${lon},${lat}&key=99749bc4ec99422ea7441072fc05cd29`;
      console.info(`WeatherService: Requesting 24h weather: ${url}`);

      const response = await httpRequest.request(url, {
        method: http.RequestMethod.GET,
        header: { 'Content-Type': 'application/json' },
        connectTimeout: 10000,
        readTimeout: 10000
      });

      console.info(`WeatherService: 24h weather response: ${response.responseCode}`);
      httpRequest.destroy();

      if (response.responseCode === 200) {
        const result: Record<string, Object> = JSON.parse(response.result as string) as Record<string, Object>;
        const code = result['code'] as string;
        
        if (code === '200' && result['hourly']) {
          const hourlyList = result['hourly'] as Object[];
          let tempMin = 100;
          let tempMax = -100;
          
          for (const item of hourlyList) {
            const hourData = item as Record<string, string>;
            const temp = parseFloat(hourData['temp'] ?? '0');
            if (temp < tempMin) tempMin = temp;
            if (temp > tempMax) tempMax = temp;
          }
          
          const tempRange: TemperatureRange = {
            tempMin: tempMin,
            tempMax: tempMax
          };
          return tempRange;
        }
      }
      return null;
    } catch (error) {
      console.error('WeatherService: 24h weather request failed:', JSON.stringify(error));
      return null;
    }
  }

  private parseWeatherData(now: Record<string, string>, locationName: string): WeatherData {
    const date = new Date();
    const updateTime = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;

    const weatherData: WeatherData = {
      temperature: parseFloat(now['temp'] ?? '0'),
      feelsLike: parseFloat(now['feelsLike'] ?? '0'),
      condition: now['text'] ?? '未知',
      conditionCode: parseInt(now['icon'] ?? '0', 10),
      humidity: parseFloat(now['humidity'] ?? '0'),
      windSpeed: parseFloat(now['windSpeed'] ?? '0'),
      windDirection: this.getWindDirectionText(parseInt(now['wind360'] ?? '0', 10)),
      visibility: parseFloat(now['vis'] ?? '0'),
      updateTime: updateTime,
      location: locationName,
      iconCode: parseInt(now['icon'] ?? '0', 10),
      minutely: [],
      tempMin: 0,
      tempMax: 0,
      precipProbability: 0
    };
    return weatherData;
  }

  private parseMinutelyData(minutelyList: Object[]): MinutelyData[] {
    const result: MinutelyData[] = [];
    console.info(`WeatherService: Parsing minutely data, count: ${minutelyList.length}`);
    for (const item of minutelyList) {
      const data = item as Record<string, string>;
      console.info(`WeatherService: Minutely item: ${JSON.stringify(data)}`);
      const minutely: MinutelyData = {
        time: data['fxTime'] ?? data['time'] ?? '',
        precipitation: parseFloat(data['precip'] ?? '0'),
        type: data['type'] ?? 'rain'
      };
      result.push(minutely);
    }
    return result;
  }

  private getDefaultWeatherData(): WeatherData {
    const date = new Date();
    const updateTime = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;

    const weatherData: WeatherData = {
      temperature: 25,
      feelsLike: 26,
      condition: '晴',
      conditionCode: 100,
      humidity: 60,
      windSpeed: 3,
      windDirection: '东南风',
      visibility: 10,
      updateTime: updateTime,
      location: '南宁',
      iconCode: 100,
      minutely: [],
      tempMin: 20,
      tempMax: 30,
      precipProbability: 0
    };
    return weatherData;
  }

  private getWindDirectionText(degrees: number): string {
    const directions: string[] = ['北风', '东北风', '东风', '东南风', '南风', '西南风', '西风', '西北风'];
    const index = Math.round(degrees / 45) % 8;
    return directions[index];
  }

  getWeatherIcon(conditionCode: number): string {
    if (conditionCode >= 100 && conditionCode <= 103) {
      return '☀️';
    } else if (conditionCode >= 104 && conditionCode <= 153) {
      return '⛅';
    } else if (conditionCode >= 200 && conditionCode <= 213) {
      return '🌬️';
    } else if (conditionCode >= 300 && conditionCode <= 313) {
      return '🌧️';
    } else if (conditionCode >= 400 && conditionCode <= 407) {
      return '🌨️';
    } else if (conditionCode >= 500 && conditionCode <= 515) {
      return '🌫️';
    } else {
      return '🌤️';
    }
  }

  getWeatherColor(conditionCode: number): string {
    if (conditionCode >= 100 && conditionCode <= 103) {
      return '#FF9500';
    } else if (conditionCode >= 300 && conditionCode <= 313) {
      return '#007AFF';
    } else if (conditionCode >= 400 && conditionCode <= 407) {
      return '#5856D6';
    } else {
      return '#8E8E93';
    }
  }

  getPrecipitationColor(precipitation: number): string {
    if (precipitation === 0) {
      return '#E5E5EA';
    } else if (precipitation < 0.1) {
      return '#B4D4E8';
    } else if (precipitation < 0.5) {
      return '#64B5F6';
    } else if (precipitation < 1.0) {
      return '#42A5F5';
    } else if (precipitation < 2.0) {
      return '#2196F3';
    } else {
      return '#1976D2';
    }
  }

  hasRainForecast(minutely: MinutelyData[]): boolean {
    if (!minutely || minutely.length === 0) {
      return false;
    }
    return minutely.some(m => m.precipitation > 0);
  }

  getRainForecastSummary(minutely: MinutelyData[]): string {
    if (!minutely || minutely.length === 0) {
      return '暂无降水预报';
    }

    const rainMinutes = minutely.filter(m => m.precipitation > 0);
    if (rainMinutes.length === 0) {
      return '未来2小时无降水';
    }

    const firstRain = rainMinutes[0];
    const rainTime = new Date(firstRain.time);
    const now = new Date();
    const minutesAway = Math.round((rainTime.getTime() - now.getTime()) / 60000);

    if (minutesAway <= 0) {
      return '正在下雨';
    } else if (minutesAway < 60) {
      return `${minutesAway}分钟后有雨`;
    } else {
      return `1小时后有雨`;
    }
  }

  clearCache(): void {
    this.cachedWeather = null;
    this.lastUpdateTime = 0;
  }
}
