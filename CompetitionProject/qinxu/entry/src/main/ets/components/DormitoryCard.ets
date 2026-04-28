/**
 * 宿舍信息卡片组件模块
 * 
 * 提供宿舍相关信息展示和管理，包括：
 * - 一卡通余额查询
 * - 电费余额查询
 * - 消费/充值记录查看
 * - 宿舍设置管理
 * 
 * @module DormitoryCard
 */

import { XyyxtApiClient, XyyxtBalanceResponse, XyyxtElectricityResponse, XyyxtBuilding, XyyxtRoom, XyyxtConsumption } from '../utils/XyyxtApiClient';
import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';
import router from '@ohos.router';

/**
 * 校区信息接口
 */
interface Campus {
  /** 校区ID */
  id: string;
  /** 校区名称 */
  name: string;
}

/**
 * 宿舍设置数据接口
 */
interface DormitorySettings {
  /** 登录账号 */
  username: string;
  /** 登录密码 */
  password: string;
  /** 房间名称 */
  roomName: string;
  /** 房间ID */
  roomId: string;
  /** 楼栋ID */
  buildingId: string;
  /** 楼栋名称 */
  buildingName: string;
  /** 校区ID */
  campusId: string;
  /** 校区名称 */
  campusName: string;
}

/**
 * 宿舍信息卡片组件
 * 
 * 展示一卡通余额、电费余额等信息，支持登录和宿舍设置
 * 
 * @example
 * ```typescript
 * DormitoryCard()
 * ```
 */
@Component
export struct DormitoryCard {
  /** 学生ID（从全局存储获取） */
  @StorageProp('studentId') studentId: string = '';
  /** 暗色模式标志 */
  @State isDarkMode: boolean = false;
  /** 显示设置面板标志 */
  @State showSettings: boolean = false;
  /** 显示消费记录面板标志 */
  @State showConsumptions: boolean = false;
  /** 宿舍设置数据 */
  @State settings: DormitorySettings = {
    username: '',
    password: '',
    roomName: '',
    roomId: '',
    buildingId: '',
    buildingName: '',
    campusId: '',
    campusName: ''
  };
  /** 一卡通余额 */
  @State cardBalance: number = 0;
  /** 电费余额 */
  @State elecBalance: number = 0;
  /** 加载状态 */
  @State isLoading: boolean = false;
  /** 登录状态 */
  @State isLoggedIn: boolean = false;
  /** 登录用户名 */
  @State loginUsername: string = '';
  /** 登录错误信息 */
  @State loginError: string = '';
  /** 初始化完成标志 */
  @State initialized: boolean = false;
  /** 消费记录列表 */
  @State consumptions: XyyxtConsumption[] = [];
  /** 充值记录列表 */
  @State recharges: XyyxtConsumption[] = [];
  /** 加载消费记录状态 */
  @State loadingConsumptions: boolean = false;
  /** 加载充值记录状态 */
  @State loadingRecharges: boolean = false;
  /** 记录Tab索引（0:充值 1:消费） */
  @State recordTab: number = 0;
  /** 设置步骤（1:登录 2:校区 3:楼栋 4:房间） */
  @State step: number = 1;
  /** 校区列表 */
  @State campuses: Campus[] = [
    { id: 'nnxq', name: '南宁校区' },
    { id: 'glxq', name: '桂林校区' }
  ];
  /** 楼栋列表 */
  @State buildings: XyyxtBuilding[] = [];
  /** 房间列表 */
  @State rooms: XyyxtRoom[] = [];
  /** 已选校区 */
  @State selectedCampus: Campus | null = null;
  /** 已选楼栋 */
  @State selectedBuilding: XyyxtBuilding | null = null;
  /** 已选房间 */
  @State selectedRoom: XyyxtRoom | null = null;
  /** 加载楼栋状态 */
  @State loadingBuildings: boolean = false;
  /** 加载房间状态 */
  @State loadingRooms: boolean = false;
  /** 一信通API客户端实例 */
  private xyyxtClient: XyyxtApiClient = XyyxtApiClient.getInstance();
  /** 偏好设置存储名称 */
  private PREFERENCES_NAME = 'dormitory_settings';

  aboutToAppear(): void {
    setTimeout(() => {
      this.loadSettings();
    }, 300);
  }

  private async loadSettings(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const dataPreferences = await preferences.getPreferences(context, this.PREFERENCES_NAME);
      const savedUsername = await dataPreferences.get('username', '') as string;
      const savedPassword = await dataPreferences.get('password', '') as string;
      const savedRoomName = await dataPreferences.get('roomName', '') as string;
      const savedRoomId = await dataPreferences.get('roomId', '') as string;
      const savedBuildingId = await dataPreferences.get('buildingId', '') as string;
      const savedBuildingName = await dataPreferences.get('buildingName', '') as string;
      const savedCampusId = await dataPreferences.get('campusId', '') as string;
      const savedCampusName = await dataPreferences.get('campusName', '') as string;

      this.settings = {
        username: savedUsername,
        password: savedPassword,
        roomName: savedRoomName,
        roomId: savedRoomId,
        buildingId: savedBuildingId,
        buildingName: savedBuildingName,
        campusId: savedCampusId,
        campusName: savedCampusName
      };

      this.initialized = true;

      if (this.studentId && savedUsername && savedPassword) {
        this.refreshData();
      }
    } catch (error) {
      console.error('DormitoryCard loadSettings error:', JSON.stringify(error));
      this.initialized = true;
    }
  }

  private async saveSettings(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const dataPreferences = await preferences.getPreferences(context, this.PREFERENCES_NAME);
      await dataPreferences.put('username', this.settings.username);
      await dataPreferences.put('password', this.settings.password);
      await dataPreferences.put('roomName', this.settings.roomName);
      await dataPreferences.put('roomId', this.settings.roomId);
      await dataPreferences.put('buildingId', this.settings.buildingId);
      await dataPreferences.put('buildingName', this.settings.buildingName);
      await dataPreferences.put('campusId', this.settings.campusId);
      await dataPreferences.put('campusName', this.settings.campusName);
      await dataPreferences.flush();

      // 同时保存到电费提醒设置的存储中，以便设置页面使用
      const electricityPrefs = await preferences.getPreferences(context, 'electricity_reminder_settings');
      await electricityPrefs.put('roomId', this.settings.roomId);
      await electricityPrefs.put('campusId', this.settings.campusId);
      await electricityPrefs.put('buildingId', this.settings.buildingId);
      await electricityPrefs.flush();
    } catch (error) {
      console.error('DormitoryCard saveSettings error:', JSON.stringify(error));
    }
  }

  async refreshData(): Promise<void> {
    if (!this.settings.username || !this.settings.password) {
      return;
    }

    this.isLoading = true;
    try {
      const loginResult = await this.xyyxtClient.login(this.settings.username, this.settings.password);
      if (!loginResult.success) {
        this.isLoggedIn = false;
        this.isLoading = false;
        return;
      }

      this.isLoggedIn = true;
      this.loginUsername = this.settings.username;

      const balanceResult: XyyxtBalanceResponse = await this.xyyxtClient.getBalance(this.settings.username);
      if (balanceResult.success && balanceResult.data !== undefined && balanceResult.data !== null) {
        const balance = balanceResult.data;
        if (typeof balance === 'number') {
          this.cardBalance = balance;
        } else if (typeof balance === 'string') {
          this.cardBalance = parseFloat(balance) || 0;
        }
      }

      if (this.settings.roomId) {
        const elecResult: XyyxtElectricityResponse = await this.xyyxtClient.getElectricity(
          this.settings.username,
          this.settings.roomId,
          this.settings.campusId,
          this.settings.buildingId
        );
        if (elecResult.success && elecResult.data !== undefined && elecResult.data !== null) {
          const elecData = elecResult.data;
          if (typeof elecData === 'object' && elecData.balance !== undefined) {
            this.elecBalance = typeof elecData.balance === 'number'
              ? elecData.balance
              : parseFloat(String(elecData.balance)) || 0;
          } else if (typeof elecData === 'number') {
            this.elecBalance = elecData;
          } else if (typeof elecData === 'string') {
            this.elecBalance = parseFloat(elecData) || 0;
          }
        }
      }
    } catch (error) {
      console.error('DormitoryCard refreshData error:', JSON.stringify(error));
    }
    this.isLoading = false;
  }

  private async loadConsumptions(): Promise<void> {
    if (!this.settings.username || !this.isLoggedIn) {
      return;
    }

    this.loadingConsumptions = true;
    try {
      const result = await this.xyyxtClient.getConsumptions(this.settings.username, '', '', 1, 20);
      if (result.success && result.data) {
        this.consumptions = result.data;
      }
    } catch (error) {
      console.error('DormitoryCard loadConsumptions error:', JSON.stringify(error));
    }
    this.loadingConsumptions = false;
  }

  private async loadRecharges(): Promise<void> {
    if (!this.settings.username || !this.isLoggedIn) {
      return;
    }

    this.loadingRecharges = true;
    try {
      const result = await this.xyyxtClient.getRecharges(this.settings.username, '', '', 1, 20);
      if (result.success && result.data) {
        this.recharges = result.data;
      }
    } catch (error) {
      console.error('DormitoryCard loadRecharges error:', JSON.stringify(error));
    }
    this.loadingRecharges = false;
  }

  private async handleLogin(): Promise<void> {
    if (!this.settings.username) {
      this.loginError = '请输入账号';
      return;
    }

    if (!this.settings.password) {
      this.loginError = '请输入密码';
      return;
    }

    this.isLoading = true;
    this.loginError = '';

    try {
      const loginResult = await this.xyyxtClient.login(this.settings.username, this.settings.password);
      if (loginResult.success) {
        this.isLoggedIn = true;
        this.loginUsername = this.settings.username;
        this.step = 2;
      } else {
        this.loginError = loginResult.message || '登录失败，请检查账号密码';
      }
    } catch (error) {
      this.loginError = '网络请求失败';
    }

    this.isLoading = false;
  }

  private async loadBuildings(): Promise<void> {
    if (!this.selectedCampus) {
      return;
    }
    this.loadingBuildings = true;
    try {
      const result = await this.xyyxtClient.getBuildings(this.settings.username, this.selectedCampus.id);
      if (result.success && result.data) {
        this.buildings = result.data;
      }
    } catch (error) {
      console.error('loadBuildings error:', JSON.stringify(error));
    }
    this.loadingBuildings = false;
  }

  private async loadRooms(buildingId: string): Promise<void> {
    if (!this.selectedCampus) {
      return;
    }
    this.loadingRooms = true;
    try {
      const result = await this.xyyxtClient.getRooms(this.settings.username, buildingId, this.selectedCampus.id);
      if (result.success && result.data) {
        this.rooms = result.data;
      }
    } catch (error) {
      console.error('loadRooms error:', JSON.stringify(error));
    }
    this.loadingRooms = false;
  }

  private async confirmSelection(): Promise<void> {
    if (this.selectedRoom) {
      this.settings.roomId = this.selectedRoom.room_id;
      this.settings.roomName = this.selectedRoom.room_name;
    }
    if (this.selectedBuilding) {
      this.settings.buildingId = this.selectedBuilding.loudong_id;
      this.settings.buildingName = this.selectedBuilding.loudong_name;
    }
    if (this.selectedCampus) {
      this.settings.campusId = this.selectedCampus.id;
      this.settings.campusName = this.selectedCampus.name;
    }

    await this.saveSettings();
    this.showSettings = false;
    this.refreshData();
  }

  private async logout(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const dataPreferences = await preferences.getPreferences(context, this.PREFERENCES_NAME);
      await dataPreferences.clear();
      await dataPreferences.flush();

      this.settings = {
        username: '',
        password: '',
        roomName: '',
        roomId: '',
        buildingId: '',
        buildingName: '',
        campusId: '',
        campusName: ''
      };
      this.isLoggedIn = false;
      this.loginUsername = '';
      this.cardBalance = 0;
      this.elecBalance = 0;
      this.showSettings = false;
    } catch (error) {
      console.error('DormitoryCard logout error:', JSON.stringify(error));
    }
  }

  @Builder
  buildConsumptionsSheet() {
    Column() {
      Row() {
        Column().width(24)

        Text('交易记录')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Column().width(24)
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 8, bottom: 12 })

      Row({ space: 8 }) {
        Button('充值记录')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(36)
          .layoutWeight(1)
          .backgroundColor(this.recordTab === 0 ? $r('app.color.primary') : $r('app.color.background_secondary'))
          .fontColor(this.recordTab === 0 ? $r('app.color.button_text') : $r('app.color.text_secondary'))
          .borderRadius(10)
          .onClick(() => {
            this.recordTab = 0;
            if (this.recharges.length === 0) {
              this.loadRecharges();
            }
          })

        Button('消费记录')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(36)
          .layoutWeight(1)
          .backgroundColor(this.recordTab === 1 ? $r('app.color.primary') : $r('app.color.background_secondary'))
          .fontColor(this.recordTab === 1 ? $r('app.color.button_text') : $r('app.color.text_secondary'))
          .borderRadius(10)
          .onClick(() => {
            this.recordTab = 1;
            if (this.consumptions.length === 0) {
              this.loadConsumptions();
            }
          })
      }
      .width('100%')
      .padding({ left: 20, right: 20, bottom: 12 })

      if (this.recordTab === 0) {
        if (this.loadingRecharges) {
          Column({ space: 12 }) {
            LoadingProgress()
              .width(48)
              .height(48)
              .color($r('app.color.primary'))
            Text('正在加载充值记录...')
              .fontSize(15)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .layoutWeight(1)
          .justifyContent(FlexAlign.Center)
        } else if (this.recharges.length === 0) {
          Column({ space: 12 }) {
            Image($r('app.media.doc_text_badge_checkmark'))
              .width(56)
              .height(56)
              .fillColor($r('app.color.text_tertiary'))
            Text('暂无充值记录')
              .fontSize(15)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .layoutWeight(1)
          .justifyContent(FlexAlign.Center)
        } else {
          List({ space: 12 }) {
            ForEach(this.recharges, (item: XyyxtConsumption, index: number) => {
              ListItem() {
                Row() {
                  Column() {
                    Image($r('app.media.arrow_counterclockwise_clock'))
                      .width(20)
                      .height(20)
                      .fillColor($r('app.color.success'))
                  }
                  .width(44)
                  .height(44)
                  .justifyContent(FlexAlign.Center)
                  .backgroundColor('#E8F8EC')
                  .borderRadius(12)

                  Column({ space: 6 }) {
                    Text(item.reason || '充值')
                      .fontSize(15)
                      .fontWeight(FontWeight.Medium)
                      .fontColor($r('app.color.text_primary'))
                      .maxLines(1)
                      .textOverflow({ overflow: TextOverflow.Ellipsis })

                    Text(item.time || '')
                      .fontSize(12)
                      .fontColor($r('app.color.text_tertiary'))
                  }
                  .alignItems(HorizontalAlign.Start)
                  .layoutWeight(1)
                  .margin({ left: 12 })

                  Column({ space: 6 }) {
                    if (item.amount !== undefined) {
                      Text(`+¥${Math.abs(item.amount).toFixed(2)}`)
                        .fontSize(17)
                        .fontWeight(FontWeight.Bold)
                        .fontColor($r('app.color.success'))
                    }

                    if (item.balance !== undefined) {
                      Text(`余额 ¥${item.balance.toFixed(2)}`)
                        .fontSize(11)
                        .fontColor($r('app.color.text_tertiary'))
                    }
                  }
                  .alignItems(HorizontalAlign.End)
                }
                .width('100%')
                .padding(16)
                .backgroundColor($r('app.color.background_secondary'))
                .borderRadius(16)
              }
            }, (item: XyyxtConsumption, index: number) => index.toString())
          }
          .width('100%')
          .layoutWeight(1)
          .padding({ left: 20, right: 20, bottom: 20 })
        }
      } else {
        if (this.loadingConsumptions) {
          Column({ space: 12 }) {
            LoadingProgress()
              .width(48)
              .height(48)
              .color($r('app.color.primary'))
            Text('正在加载消费记录...')
              .fontSize(15)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .layoutWeight(1)
          .justifyContent(FlexAlign.Center)
        } else if (this.consumptions.length === 0) {
          Column({ space: 12 }) {
            Image($r('app.media.doc_text_badge_checkmark'))
              .width(56)
              .height(56)
              .fillColor($r('app.color.text_tertiary'))
            Text('暂无消费记录')
              .fontSize(15)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .layoutWeight(1)
          .justifyContent(FlexAlign.Center)
        } else {
          List({ space: 12 }) {
            ForEach(this.consumptions, (item: XyyxtConsumption, index: number) => {
              ListItem() {
                Row() {
                  Column() {
                    Image($r('app.media.staroflife_rectangle'))
                      .width(20)
                      .height(20)
                      .fillColor($r('app.color.primary'))
                  }
                  .width(44)
                  .height(44)
                  .justifyContent(FlexAlign.Center)
                  .backgroundColor($r('app.color.info_light'))
                  .borderRadius(12)

                  Column({ space: 6 }) {
                    Text(item.reason || '消费')
                      .fontSize(15)
                      .fontWeight(FontWeight.Medium)
                      .fontColor($r('app.color.text_primary'))
                      .maxLines(1)
                      .textOverflow({ overflow: TextOverflow.Ellipsis })

                    Text(item.time || '')
                      .fontSize(12)
                      .fontColor($r('app.color.text_tertiary'))
                  }
                  .alignItems(HorizontalAlign.Start)
                  .layoutWeight(1)
                  .margin({ left: 12 })

                  Column({ space: 6 }) {
                    if (item.amount !== undefined) {
                      Text(`-¥${Math.abs(item.amount).toFixed(2)}`)
                        .fontSize(17)
                        .fontWeight(FontWeight.Bold)
                        .fontColor($r('app.color.danger'))
                    }

                    if (item.balance !== undefined) {
                      Text(`余额 ¥${item.balance.toFixed(2)}`)
                        .fontSize(11)
                        .fontColor($r('app.color.text_tertiary'))
                    }
                  }
                  .alignItems(HorizontalAlign.End)
                }
                .width('100%')
                .padding(16)
                .backgroundColor($r('app.color.background_secondary'))
                .borderRadius(16)
              }
            }, (item: XyyxtConsumption, index: number) => index.toString())
          }
          .width('100%')
          .layoutWeight(1)
          .padding({ left: 20, right: 20, bottom: 20 })
        }
      }
    }
    .width('100%')
    .height('100%')
  }

  @Builder
  buildLoginStep() {
    if (this.isLoggedIn) {
      Column({ space: 20 }) {
        Column({ space: 12 }) {
          Text('当前账号')
            .fontSize(13)
            .fontColor($r('app.color.text_tertiary'))
            .width('100%')

          Row() {
            Column() {
              Image($r('app.media.person'))
                .width(28)
                .height(28)
                .fillColor($r('app.color.primary'))
            }
            .width(48)
            .height(48)
            .justifyContent(FlexAlign.Center)
            .backgroundColor($r('app.color.info_light'))
            .borderRadius(14)

            Column({ space: 4 }) {
              Text(this.loginUsername)
                .fontSize(16)
                .fontWeight(FontWeight.Medium)
                .fontColor($r('app.color.text_primary'))

              Row({ space: 4 }) {
                Image($r('app.media.checkmark_circle'))
                  .width(14)
                  .height(14)
                  .fillColor($r('app.color.success'))

                Text('已登录')
                  .fontSize(12)
                  .fontColor($r('app.color.success'))
              }
            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1)
            .margin({ left: 12 })
          }
          .width('100%')
          .padding(16)
          .backgroundColor($r('app.color.background_secondary'))
          .borderRadius(16)
        }
        .width('100%')

        if (this.settings.roomName) {
          Column({ space: 12 }) {
            Text('当前宿舍')
              .fontSize(13)
              .fontColor($r('app.color.text_tertiary'))
              .width('100%')

            Row() {
              Column() {
                Image($r('app.media.house_fill'))
                  .width(28)
                  .height(28)
                  .fillColor($r('app.color.primary'))
              }
              .width(48)
              .height(48)
              .justifyContent(FlexAlign.Center)
              .backgroundColor($r('app.color.info_light'))
              .borderRadius(14)

              Column({ space: 4 }) {
                Text(this.settings.roomName)
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))

                if (this.settings.buildingName) {
                  Text(this.settings.buildingName)
                    .fontSize(12)
                    .fontColor($r('app.color.text_tertiary'))
                }
                if (this.settings.campusName) {
                  Text(this.settings.campusName)
                    .fontSize(11)
                    .fontColor($r('app.color.text_tertiary'))
                }
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
              .margin({ left: 12 })
            }
            .width('100%')
            .padding(16)
            .backgroundColor($r('app.color.background_secondary'))
            .borderRadius(16)
          }
          .width('100%')
        }

        Blank()

        Button('重新选择宿舍')
          .width('100%')
          .height(50)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .backgroundColor($r('app.color.primary'))
          .fontColor($r('app.color.button_text'))
          .borderRadius(14)
          .onClick(() => {
            this.step = 2;
          })

        Button('退出登录')
          .width('100%')
          .height(50)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .backgroundColor($r('app.color.background_secondary'))
          .fontColor($r('app.color.danger'))
          .borderRadius(14)
          .margin({ top: 12 })
          .onClick(() => {
            this.logout();
          })
      }
      .width('100%')
      .padding({ left: 20, right: 20, bottom: 30 })
      .layoutWeight(1)
    } else {
      Column({ space: 20 }) {
        Column({ space: 12 }) {
          Text('账号')
            .fontSize(13)
            .fontColor($r('app.color.text_tertiary'))
            .width('100%')

          TextInput({ placeholder: '请输入手机号', text: $$this.settings.username })
            .type(InputType.Normal)
            .maxLength(11)
            .width('100%')
            .height(52)
            .fontSize(16)
            .backgroundColor($r('app.color.background_secondary'))
            .borderRadius(14)
            .padding({ left: 16, right: 16 })
        }
        .width('100%')

        Column({ space: 12 }) {
          Text('密码')
            .fontSize(13)
            .fontColor($r('app.color.text_tertiary'))
            .width('100%')

          TextInput({ placeholder: '请输入密码', text: $$this.settings.password })
            .type(InputType.Password)
            .width('100%')
            .height(52)
            .fontSize(16)
            .backgroundColor($r('app.color.background_secondary'))
            .borderRadius(14)
            .padding({ left: 16, right: 16 })
        }
        .width('100%')

        if (this.loginError) {
          Row({ space: 8 }) {
            Image($r('app.media.exclamationmark_triangle_fill'))
              .width(16)
              .height(16)
              .fillColor($r('app.color.danger'))

            Text(this.loginError)
              .fontSize(13)
              .fontColor($r('app.color.danger'))
          }
          .width('100%')
          .padding(12)
          .backgroundColor($r('app.color.danger_light'))
          .borderRadius(10)
        }

        Blank()

        Button(this.isLoading ? '登录中...' : '登录')
          .width('100%')
          .height(50)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .backgroundColor($r('app.color.primary'))
          .fontColor($r('app.color.button_text'))
          .borderRadius(14)
          .enabled(!this.isLoading)
          .onClick(() => {
            this.handleLogin();
          })
      }
      .width('100%')
      .padding({ left: 20, right: 20, bottom: 30 })
      .layoutWeight(1)
    }
  }

  @Builder
  buildCampusStep() {
    Column() {
      Column({ space: 8 }) {
        Text('请选择您所在的校区')
          .fontSize(14)
          .fontColor($r('app.color.text_tertiary'))
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 8, bottom: 12 })

      List({ space: 12 }) {
        ForEach(this.campuses, (campus: Campus) => {
          ListItem() {
            Row() {
              Column() {
                Image($r('app.media.house_fill'))
                  .width(24)
                  .height(24)
                  .fillColor(this.selectedCampus?.id === campus.id ?
                    $r('app.color.primary') : $r('app.color.text_secondary'))
              }
              .width(48)
              .height(48)
              .justifyContent(FlexAlign.Center)
              .backgroundColor(this.selectedCampus?.id === campus.id ?
                $r('app.color.info_light') : $r('app.color.background_secondary'))
              .borderRadius(14)

              Text(campus.name)
                .fontSize(16)
                .fontWeight(FontWeight.Medium)
                .fontColor($r('app.color.text_primary'))
                .layoutWeight(1)
                .margin({ left: 12 })

              if (this.selectedCampus?.id === campus.id) {
                Image($r('app.media.checkmark_circle'))
                  .width(24)
                  .height(24)
                  .fillColor($r('app.color.primary'))
              } else {
                Column()
                  .width(24)
                  .height(24)
                  .borderWidth(2)
                  .borderColor($r('app.color.divider'))
                  .borderRadius(12)
              }
            }
            .width('100%')
            .padding(16)
            .backgroundColor(this.selectedCampus?.id === campus.id ?
              $r('app.color.info_light') : $r('app.color.background_secondary'))
            .borderRadius(16)
            .border({
              width: this.selectedCampus?.id === campus.id ? 2 : 0,
              color: $r('app.color.primary')
            })
            .onClick(() => {
              this.selectedCampus = campus;
            })
          }
        })
      }
      .width('100%')
      .layoutWeight(1)
      .padding({ left: 20, right: 20 })

      Button('下一步')
        .width('100%')
        .height(50)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.primary'))
        .fontColor($r('app.color.button_text'))
        .borderRadius(14)
        .enabled(this.selectedCampus !== null)
        .margin({ top: 16, bottom: 30, left: 20, right: 20 })
        .onClick(async () => {
          if (this.selectedCampus) {
            this.step = 3;
            await this.loadBuildings();
          }
        })
    }
    .width('100%')
    .layoutWeight(1)
  }

  @Builder
  buildBuildingStep() {
    Column() {
      Column({ space: 8 }) {
        Text('请选择您所在的楼栋')
          .fontSize(14)
          .fontColor($r('app.color.text_tertiary'))
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 8, bottom: 12 })

      if (this.loadingBuildings) {
        Column({ space: 12 }) {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'))
          Text('正在加载楼栋列表...')
            .fontSize(15)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else if (this.buildings.length === 0) {
        Column({ space: 12 }) {
          Image($r('app.media.house_fill'))
            .width(56)
            .height(56)
            .fillColor($r('app.color.text_tertiary'))
          Text('暂无楼栋数据')
            .fontSize(15)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else {
        List({ space: 12 }) {
          ForEach(this.buildings, (building: XyyxtBuilding) => {
            ListItem() {
              Row() {
                Column() {
                  Image($r('app.media.house_fill'))
                    .width(24)
                    .height(24)
                    .fillColor(this.selectedBuilding?.loudong_id === building.loudong_id ?
                      $r('app.color.primary') : $r('app.color.text_secondary'))
                }
                .width(48)
                .height(48)
                .justifyContent(FlexAlign.Center)
                .backgroundColor(this.selectedBuilding?.loudong_id === building.loudong_id ?
                  $r('app.color.info_light') : $r('app.color.background_secondary'))
                .borderRadius(14)

                Text(building.loudong_name)
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .layoutWeight(1)
                  .margin({ left: 12 })

                if (this.selectedBuilding?.loudong_id === building.loudong_id) {
                  Image($r('app.media.checkmark_circle'))
                    .width(24)
                    .height(24)
                    .fillColor($r('app.color.primary'))
                } else {
                  Column()
                    .width(24)
                    .height(24)
                    .borderWidth(2)
                    .borderColor($r('app.color.divider'))
                    .borderRadius(12)
                }
              }
              .width('100%')
              .padding(16)
              .backgroundColor(this.selectedBuilding?.loudong_id === building.loudong_id ?
                $r('app.color.info_light') : $r('app.color.background_secondary'))
              .borderRadius(16)
              .border({
                width: this.selectedBuilding?.loudong_id === building.loudong_id ? 2 : 0,
                color: $r('app.color.primary')
              })
              .onClick(() => {
                this.selectedBuilding = building;
              })
            }
          })
        }
        .width('100%')
        .layoutWeight(1)
        .padding({ left: 20, right: 20 })
      }

      Button('下一步')
        .width('100%')
        .height(50)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.primary'))
        .fontColor($r('app.color.button_text'))
        .borderRadius(14)
        .enabled(this.selectedBuilding !== null)
        .margin({ top: 16, bottom: 30, left: 20, right: 20 })
        .onClick(async () => {
          if (this.selectedBuilding) {
            this.step = 4;
            await this.loadRooms(this.selectedBuilding.loudong_id);
          }
        })
    }
    .width('100%')
    .layoutWeight(1)
  }

  @Builder
  buildRoomStep() {
    Column() {
      Column({ space: 8 }) {
        Text('请选择您的房间')
          .fontSize(14)
          .fontColor($r('app.color.text_tertiary'))
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 8, bottom: 12 })

      if (this.loadingRooms) {
        Column({ space: 12 }) {
          LoadingProgress()
            .width(48)
            .height(48)
            .color($r('app.color.primary'))
          Text('正在加载房间列表...')
            .fontSize(15)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else if (this.rooms.length === 0) {
        Column({ space: 12 }) {
          Image($r('app.media.house_fill'))
            .width(56)
            .height(56)
            .fillColor($r('app.color.text_tertiary'))
          Text('暂无房间数据')
            .fontSize(15)
            .fontColor($r('app.color.text_secondary'))
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
      } else {
        List({ space: 12 }) {
          ForEach(this.rooms, (room: XyyxtRoom) => {
            ListItem() {
              Row() {
                Column() {
                  Image($r('app.media.house'))
                    .width(24)
                    .height(24)
                    .fillColor(this.selectedRoom?.room_id === room.room_id ?
                      $r('app.color.primary') : $r('app.color.text_secondary'))
                }
                .width(48)
                .height(48)
                .justifyContent(FlexAlign.Center)
                .backgroundColor(this.selectedRoom?.room_id === room.room_id ?
                  $r('app.color.info_light') : $r('app.color.background_secondary'))
                .borderRadius(14)

                Text(room.room_name)
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .layoutWeight(1)
                  .margin({ left: 12 })

                if (this.selectedRoom?.room_id === room.room_id) {
                  Image($r('app.media.checkmark_circle'))
                    .width(24)
                    .height(24)
                    .fillColor($r('app.color.primary'))
                } else {
                  Column()
                    .width(24)
                    .height(24)
                    .borderWidth(2)
                    .borderColor($r('app.color.divider'))
                    .borderRadius(12)
                }
              }
              .width('100%')
              .padding(16)
              .backgroundColor(this.selectedRoom?.room_id === room.room_id ?
                $r('app.color.info_light') : $r('app.color.background_secondary'))
              .borderRadius(16)
              .border({
                width: this.selectedRoom?.room_id === room.room_id ? 2 : 0,
                color: $r('app.color.primary')
              })
              .onClick(() => {
                this.selectedRoom = room;
              })
            }
          })
        }
        .width('100%')
        .layoutWeight(1)
        .padding({ left: 20, right: 20 })
      }

      Button('确认选择')
        .width('100%')
        .height(50)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.primary'))
        .fontColor($r('app.color.button_text'))
        .borderRadius(14)
        .enabled(this.selectedRoom !== null)
        .margin({ top: 16, bottom: 30, left: 20, right: 20 })
        .onClick(() => {
          this.confirmSelection();
        })
    }
    .width('100%')
    .layoutWeight(1)
  }

  @Builder
  buildSettingsSheet() {
    Column() {
      Row() {
        if (this.step > 1) {
          Row() {
            Image($r('app.media.chevron_left'))
              .width(20)
              .height(20)
              .fillColor($r('app.color.text_primary'))
          }
          .width(36)
          .height(36)
          .justifyContent(FlexAlign.Center)
          .backgroundColor($r('app.color.background_secondary'))
          .borderRadius(10)
          .onClick(() => {
            this.step = this.step - 1;
          })
        } else {
          Column().width(36).height(36)
        }

        Text(this.step === 1 ? (this.isLoggedIn ? '宿舍设置' : '登录校园一信通') :
          this.step === 2 ? '选择校区' :
            this.step === 3 ? '选择楼栋' : '选择房间')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Column().width(36).height(36)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 8, bottom: 16 })

      if (this.step === 1) {
        this.buildLoginStep()
      } else if (this.step === 2) {
        this.buildCampusStep()
      } else if (this.step === 3) {
        this.buildBuildingStep()
      } else if (this.step === 4) {
        this.buildRoomStep()
      }
    }
    .width('100%')
    .height('100%')
  }

  build() {
    Column() {
      Column() {
        Row() {
          Row({ space: 6 }) {
            Image($r('app.media.house_fill'))
              .width(18)
              .height(18)
              .fillColor($r('app.color.primary'))

            Text('宿舍信息')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor($r('app.color.text_primary'))
          }

          Blank()

          Row({ space: 8 }) {
            if (this.isLoggedIn) {
              Row() {
                Image($r('app.media.staroflife_rectangle'))
                  .width(18)
                  .height(18)
                  .fillColor($r('app.color.text_secondary'))
              }
              .width(36)
              .height(36)
              .justifyContent(FlexAlign.Center)
              .borderRadius(8)
              .backgroundColor($r('app.color.background_secondary'))
              .onClick(() => {
                this.recordTab = 0;
                this.loadRecharges();
                this.showConsumptions = true;
              })

              Row() {
                Image($r('app.media.arrow_clockwise'))
                  .width(18)
                  .height(18)
                  .fillColor($r('app.color.text_secondary'))
              }
              .width(36)
              .height(36)
              .justifyContent(FlexAlign.Center)
              .borderRadius(8)
              .backgroundColor($r('app.color.background_secondary'))
              .onClick(() => {
                if (!this.isLoading) {
                  this.refreshData();
                }
              })

              Row() {
                Image($r('app.media.bell_fill'))
                  .width(18)
                  .height(18)
                  .fillColor($r('app.color.text_secondary'))
              }
              .width(36)
              .height(36)
              .justifyContent(FlexAlign.Center)
              .borderRadius(8)
              .backgroundColor($r('app.color.background_secondary'))
              .onClick(() => {
                router.pushUrl({
                  url: 'pages/ElectricityReminderSettingsPage'
                });
              })
            }

            Row() {
              Image($r('app.media.gearshape'))
                .width(18)
                .height(18)
                .fillColor($r('app.color.text_secondary'))
            }
            .width(36)
            .height(36)
            .justifyContent(FlexAlign.Center)
            .borderRadius(8)
            .backgroundColor($r('app.color.background_secondary'))
            .onClick(() => {
              this.step = 1;
              this.loginError = '';
              this.selectedCampus = null;
              this.selectedBuilding = null;
              this.selectedRoom = null;
              this.showSettings = true;
            })
          }
        }
        .width('100%')
        .margin({ bottom: 12 })

        if (!this.studentId) {
          Column({ space: 8 }) {
            Image($r('app.media.house_fill'))
              .width(40)
              .height(40)
              .fillColor($r('app.color.text_tertiary'))

            Text('请先登录教务系统')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .padding({ top: 12, bottom: 12 })
        } else if (this.isLoading) {
          Row() {
            LoadingProgress()
              .width(20)
              .height(20)
              .color($r('app.color.primary'))
            Text('加载中...')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
              .margin({ left: 8 })
          }
          .width('100%')
          .justifyContent(FlexAlign.Center)
          .padding({ top: 20, bottom: 20 })
        } else if (!this.isLoggedIn) {
          Column({ space: 8 }) {
            Image($r('app.media.house_fill'))
              .width(40)
              .height(40)
              .fillColor($r('app.color.text_tertiary'))

            Text('点击设置按钮登录校园一信通')
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
          }
          .width('100%')
          .padding({ top: 12, bottom: 12 })
        } else {
          Row() {
            Column({ space: 4 }) {
              Text('一卡通余额')
                .fontSize(12)
                .fontColor($r('app.color.text_secondary'))

              Row({ space: 4 }) {
                Text(`¥${this.cardBalance.toFixed(2)}`)
                  .fontSize(22)
                  .fontWeight(FontWeight.Bold)
                  .fontColor(this.cardBalance < 20 ? $r('app.color.danger') : $r('app.color.success'))
              }

              Text(this.loginUsername)
                .fontSize(11)
                .fontColor($r('app.color.text_tertiary'))
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1)
            .padding(8)
            .borderRadius(8)

            if (this.settings.roomName) {
              Column()
                .width(1)
                .height(50)
                .backgroundColor($r('app.color.divider'))

              Column({ space: 4 }) {
                Text('电费余额')
                  .fontSize(12)
                  .fontColor($r('app.color.text_secondary'))

                Row({ space: 4 }) {
                  Text(`¥${this.elecBalance.toFixed(2)}`)
                    .fontSize(22)
                    .fontWeight(FontWeight.Bold)
                    .fontColor(this.elecBalance < 10 ? $r('app.color.danger') : $r('app.color.primary'))
                }

                Text(this.settings.roomName)
                  .fontSize(11)
                  .fontColor($r('app.color.text_tertiary'))
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
              .margin({ left: 16 })
              .padding(8)
              .borderRadius(8)
            }
          }
          .width('100%')
        }
      }
      .width('100%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(16)
      .bindSheet($$this.showSettings, this.buildSettingsSheet, {
        height: '80%',
        backgroundColor: $r('app.color.background_primary'),
        dragBar: true
      })

      Column()
        .width(0)
        .height(0)
        .bindSheet($$this.showConsumptions, this.buildConsumptionsSheet, {
          height: '70%',
          backgroundColor: $r('app.color.background_primary'),
          dragBar: true
        })
    }
  }
}