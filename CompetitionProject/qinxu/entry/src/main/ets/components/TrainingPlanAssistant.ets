/**
 * 培养计划助手组件模块
 * 
 * 提供AI驱动的课程学习规划功能，包括：
 * - 培养计划分析
 * - 学习计划生成
 * - 课程学习建议
 * - 聊天历史管理
 * 
 * @module TrainingPlanAssistant
 */

import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { pasteboard } from '@kit.BasicServicesKit';
import { AIService, AIResponse, AIModel, getModelDisplayName, getAllModels, TrainingCourseData } from '../utils/AIService';
import { SemesterUtils, SemesterInfo } from '../utils/SemesterUtils';

/**
 * 聊天消息项接口
 */
interface ChatMessageItem {
  /** 消息角色（user/assistant） */
  role: string;
  /** 消息内容 */
  content: string;
}

/**
 * 聊天历史接口
 */
interface ChatHistory {
  /** 历史记录ID */
  id: string;
  /** 对话标题 */
  title: string;
  /** 消息列表 */
  messages: ChatMessageItem[];
  /** 创建时间戳 */
  createdAt: number;
  /** 使用的AI模型 */
  model: AIModel;
}

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'training_plan_preferences';
/** AI API Key存储键名 */
const KEY_AI_API_KEY = 'ai_api_key';
/** AI模型存储键名 */
const KEY_AI_MODEL = 'ai_model';
/** 聊天历史存储键名 */
const KEY_CHAT_HISTORIES = 'chat_histories';

/**
 * 培养计划助手组件
 * 
 * 提供AI驱动的课程学习和培养计划分析功能
 * 
 * @example
 * ```typescript
 * TrainingPlanAssistant({
 *   courses: courseList,
 *   major: '计算机科学与技术',
 *   enrollmentYear: '2022',
 *   currentSemesterInfo: semesterInfo
 * })
 * ```
 */
@Entry
@Component
export struct TrainingPlanAssistant {
  /** 培养计划课程数据 */
  @Prop courses: TrainingCourseData[] = [];
  /** 专业名称 */
  @Prop major: string = '';
  /** 入学年份 */
  @Prop enrollmentYear: string = '';
  /** 当前学期信息 */
  @Prop currentSemesterInfo: SemesterInfo | null = null;
  /** 聊天消息列表 */
  @State messages: ChatMessageItem[] = [];
  /** 输入框文本 */
  @State inputText: string = '';
  /** 加载状态 */
  @State isLoading: boolean = false;
  /** 显示配置面板标志 */
  @State showConfig: boolean = false;
  /** 显示历史记录面板标志 */
  @State showHistory: boolean = false;
  /** 聊天历史记录列表 */
  @State chatHistories: ChatHistory[] = [];
  /** 当前聊天ID */
  @State currentChatId: string = '';
  /** API Key（全局存储绑定） */
  @StorageLink('ai_api_key') apiKey: string = '';
  /** 选中的AI模型（全局存储绑定） */
  @StorageLink('ai_model') selectedModel: AIModel = 'MiniMax-M2.5';
  /** AI服务实例 */
  private aiService: AIService = AIService.getInstance();
  /** 滚动控制器 */
  private scroller: Scroller = new Scroller();
  /** 应用上下文 */
  private context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;

  aboutToAppear() {
    this.loadSettings();
    this.loadChatHistories();
  }

  /**
   * 加载设置
   * 从本地存储加载API配置
   */
  private async loadSettings(): Promise<void> {
    try {
      const prefs = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      const savedApiKey = await prefs.get(KEY_AI_API_KEY, '') as string;
      const savedModel = await prefs.get(KEY_AI_MODEL, 'MiniMax-M2.5') as string;
      
      if (savedApiKey) {
        this.apiKey = savedApiKey;
        AppStorage.SetOrCreate('ai_api_key', savedApiKey);
      }
      if (savedModel) {
        this.selectedModel = savedModel as AIModel;
        AppStorage.SetOrCreate('ai_model', savedModel);
      }
      
      this.aiService.setConfig({
        apiKey: this.apiKey,
        model: this.selectedModel
      });
    } catch (error) {
      console.error('加载设置失败:', JSON.stringify(error));
    }
  }

  /**
   * 加载聊天历史
   * 从本地存储加载历史对话记录
   */
  private async loadChatHistories(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      const historiesJson = await dataPreferences.get(KEY_CHAT_HISTORIES, '[]') as string;
      this.chatHistories = JSON.parse(historiesJson) as ChatHistory[];
      
      if (this.chatHistories.length > 0) {
        const lastChat = this.chatHistories[0];
        this.currentChatId = lastChat.id;
        this.messages = [...lastChat.messages];
        this.aiService.setHistoryFromMessages(lastChat.messages);
      }
    } catch (error) {
      console.error('加载历史记录失败:', JSON.stringify(error));
      this.chatHistories = [];
    }
  }

  /**
   * 保存聊天历史
   * 将历史记录持久化到本地存储
   */
  private async saveChatHistories(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      await dataPreferences.put(KEY_CHAT_HISTORIES, JSON.stringify(this.chatHistories));
      await dataPreferences.flush();
    } catch (error) {
      console.error('保存历史记录失败:', JSON.stringify(error));
    }
  }

  /**
   * 生成聊天ID
   * 
   * @returns 唯一的聊天ID字符串
   */
  private generateChatId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  /**
   * 获取聊天标题
   * 使用第一条用户消息作为标题
   * 
   * @returns 聊天标题字符串
   */
  private getChatTitle(): string {
    if (this.messages.length > 0) {
      const firstUserMsg = this.messages.find((m: ChatMessageItem) => m.role === 'user');
      if (firstUserMsg) {
        return firstUserMsg.content.substring(0, 20) + (firstUserMsg.content.length > 20 ? '...' : '');
      }
    }
    return '新对话';
  }

  /**
   * 保存当前聊天
   * 将当前对话保存到历史记录
   */
  private async saveCurrentChat(): Promise<void> {
    if (this.messages.length === 0) return;
    
    const now = Date.now();
    if (this.currentChatId === '') {
      this.currentChatId = this.generateChatId();
      const newHistory: ChatHistory = {
        id: this.currentChatId,
        title: this.getChatTitle(),
        messages: [...this.messages],
        createdAt: now,
        model: this.selectedModel
      };
      this.chatHistories.unshift(newHistory);
    } else {
      const index = this.chatHistories.findIndex((h: ChatHistory) => h.id === this.currentChatId);
      if (index >= 0) {
        this.chatHistories[index].messages = [...this.messages];
        this.chatHistories[index].title = this.getChatTitle();
        this.chatHistories[index].model = this.selectedModel;
      }
    }
    await this.saveChatHistories();
  }

  /**
   * 开始新对话
   * 清空当前消息并创建新的对话
   */
  private async newChat(): Promise<void> {
    await this.saveCurrentChat();
    this.currentChatId = '';
    this.messages = [];
    this.aiService.clearHistory();
    this.showHistory = false;
  }

  /**
   * 加载历史聊天
   * 
   * @param history - 要加载的聊天历史
   */
  private loadChat(history: ChatHistory): void {
    this.currentChatId = history.id;
    this.messages = [...history.messages];
    this.selectedModel = history.model;
    this.aiService.setHistoryFromMessages(history.messages);
    this.showHistory = false;
  }

  /**
   * 删除聊天历史
   * 
   * @param id - 要删除的聊天ID
   */
  private async deleteChat(id: string): Promise<void> {
    this.chatHistories = this.chatHistories.filter((h: ChatHistory) => h.id !== id);
    await this.saveChatHistories();
    
    if (this.currentChatId === id) {
      this.currentChatId = '';
      this.messages = [];
      this.aiService.clearHistory();
    }
  }

  /**
   * 保存设置
   * 将API配置持久化到本地存储
   */
  private async saveSettings(): Promise<void> {
    AppStorage.SetOrCreate('ai_api_key', this.apiKey);
    AppStorage.SetOrCreate('ai_model', this.selectedModel);
    
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      await dataPreferences.put(KEY_AI_API_KEY, this.apiKey);
      await dataPreferences.put(KEY_AI_MODEL, this.selectedModel);
      await dataPreferences.flush();
      
      this.aiService.setConfig({
        apiKey: this.apiKey,
        model: this.selectedModel
      });
      this.showConfig = false;
    } catch (error) {
      console.error('保存设置失败:', JSON.stringify(error));
    }
  }

  /**
   * 复制文本到剪贴板
   * 
   * @param text - 要复制的文本
   */
  private copyToClipboard(text: string): void {
    const pasteboardData = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, text);
    const systemPasteboard = pasteboard.getSystemPasteboard();
    systemPasteboard.setData(pasteboardData);
  }

  /**
   * 格式化时间显示
   * 将时间戳转换为友好的时间描述
   * 
   * @param timestamp - 时间戳
   * @returns 格式化后的时间字符串
   */
  private formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) {
      return '刚刚';
    } else if (diff < 3600000) {
      return Math.floor(diff / 60000) + '分钟前';
    } else if (diff < 86400000) {
      return Math.floor(diff / 3600000) + '小时前';
    } else if (diff < 604800000) {
      return Math.floor(diff / 86400000) + '天前';
    } else {
      return date.toLocaleDateString();
    }
  }

  /**
   * 开始计划分析
   * 分析培养计划并生成学习建议
   */
  private async startPlanAnalysis(): Promise<void> {
    if (this.courses.length === 0) {
      const errorMsg: ChatMessageItem = { role: 'assistant', content: '暂无培养计划数据可供分析。' };
      this.messages.push(errorMsg);
      return;
    }

    this.isLoading = true;
    const userMsg: ChatMessageItem = { role: 'user', content: '生成本学期学习计划' };
    this.messages.push(userMsg);
    
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
    }, 50);

    try {
      const response: AIResponse = await this.aiService.analyzeTrainingPlan(
        this.courses,
        this.major,
        this.enrollmentYear,
        this.currentSemesterInfo
      );
      if (response.success) {
        const assistantMsg: ChatMessageItem = { role: 'assistant', content: response.content };
        this.messages.push(assistantMsg);
      } else {
        const errorMsg: ChatMessageItem = { role: 'assistant', content: `分析失败: ${response.error}` };
        this.messages.push(errorMsg);
      }
      await this.saveCurrentChat();
      
      setTimeout(() => {
        this.scroller.scrollEdge(Edge.Bottom);
      }, 50);
    } catch (error) {
      const errorMsg: ChatMessageItem = { role: 'assistant', content: `分析失败: ${JSON.stringify(error)}` };
      this.messages.push(errorMsg);
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * 发送消息
   * 将用户消息发送给AI并获取回复
   */
  private async sendMessage(): Promise<void> {
    if (!this.inputText.trim() || this.isLoading) return;
    if (!this.apiKey) {
      this.showConfig = true;
      return;
    }

    const userMessage: string = this.inputText.trim();
    this.inputText = '';
    
    const userMsg: ChatMessageItem = { role: 'user', content: userMessage };
    this.messages.push(userMsg);
    this.isLoading = true;
    
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
    }, 50);

    try {
      const response: AIResponse = await this.aiService.analyzeTrainingPlan(
        this.courses,
        this.major,
        this.enrollmentYear,
        this.currentSemesterInfo,
        userMessage
      );
      
      if (response.success) {
        const assistantMsg: ChatMessageItem = { role: 'assistant', content: response.content };
        this.messages.push(assistantMsg);
      } else {
        const errorMsg: ChatMessageItem = { role: 'assistant', content: `错误: ${response.error}` };
        this.messages.push(errorMsg);
      }
      await this.saveCurrentChat();
      
      setTimeout(() => {
        this.scroller.scrollEdge(Edge.Bottom);
      }, 50);
    } catch (error) {
      const errorMsg: ChatMessageItem = { role: 'assistant', content: `请求失败: ${JSON.stringify(error)}` };
      this.messages.push(errorMsg);
    } finally {
      this.isLoading = false;
    }
  }

  build() {
    Column() {
      Row() {
        Button({ type: ButtonType.Circle }) {
          Image($r('app.media.ellipsis_bubble'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_secondary'))
        }
        .width(32)
        .height(32)
        .backgroundColor('transparent')
        .onClick(() => {
          this.showHistory = !this.showHistory;
        })

        Text('课程学习助手')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

        Button({ type: ButtonType.Circle }) {
          Image($r('app.media.gearshape'))
            .width(20)
            .height(20)
            .fillColor($r('app.color.text_secondary'))
        }
        .width(32)
        .height(32)
        .backgroundColor('transparent')
        .onClick(() => {
          this.showConfig = !this.showConfig;
        })
      }
      .width('100%')
      .padding({ left: 8, right: 8, top: 12, bottom: 12 })

      if (this.showHistory) {
        this.buildHistoryPanel()
      } else if (this.showConfig) {
        this.buildConfigPanel()
      } else {
        this.buildChatPanel()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_primary'))
  }

  @Builder
  buildHistoryPanel() {
    Column() {
      Row() {
        Text('聊天记录')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))

        Blank()

        Button({ type: ButtonType.Circle }) {
          Image($r('app.media.message'))
            .width(18)
            .height(18)
            .fillColor($r('app.color.primary'))
        }
        .width(32)
        .height(32)
        .backgroundColor($r('app.color.background_tertiary'))
        .onClick(() => {
          this.newChat();
        })
      }
      .width('100%')

      if (this.chatHistories.length === 0) {
        Column() {
          Text('暂无聊天记录')
            .fontSize(14)
            .fontColor($r('app.color.text_tertiary'))
            .margin({ top: 40 })
        }
        .width('100%')
        .height(200)
      } else {
        List({ space: 8 }) {
          ForEach(this.chatHistories, (history: ChatHistory) => {
            ListItem() {
              Row() {
                Column() {
                  Text(history.title)
                    .fontSize(14)
                    .fontColor($r('app.color.text_primary'))
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })

                  Text(this.formatTime(history.createdAt))
                    .fontSize(11)
                    .fontColor($r('app.color.text_tertiary'))
                    .margin({ top: 4 })
                }
                .alignItems(HorizontalAlign.Start)
                .layoutWeight(1)

                Button({ type: ButtonType.Circle }) {
                  Image($r('app.media.trash_copy'))
                    .width(16)
                    .height(16)
                    .fillColor($r('app.color.text_tertiary'))
                }
                .width(28)
                .height(28)
                .backgroundColor('transparent')
                .onClick(() => {
                  this.deleteChat(history.id);
                })
              }
              .width('100%')
              .padding(12)
              .backgroundColor(this.currentChatId === history.id ? 
                $r('app.color.background_tertiary') : 'transparent')
              .borderRadius(8)
              .onClick(() => {
                this.loadChat(history);
              })
            }
          })
        }
        .width('100%')
        .layoutWeight(1)
      }
    }
    .width('100%')
    .padding(16)
    .layoutWeight(1)
  }

  @Builder
  buildConfigPanel() {
    Column({ space: 12 }) {
      Text('API 配置')
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))

      Column({ space: 8 }) {
        Text('选择模型')
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))

        Flex({ wrap: FlexWrap.Wrap, justifyContent: FlexAlign.Start }) {
          ForEach(getAllModels(), (model: AIModel) => {
            Button(getModelDisplayName(model))
              .fontSize(12)
              .height(32)
              .margin({ right: 8, bottom: 8 })
              .backgroundColor(this.selectedModel === model ? 
                $r('app.color.primary') : $r('app.color.background_tertiary'))
              .fontColor(this.selectedModel === model ? 
                '#FFFFFF' : $r('app.color.text_primary'))
              .onClick(() => {
                this.selectedModel = model;
              })
          })
        }
        .width('100%')
      }
      .width('100%')

      Column({ space: 8 }) {
        Text('API Key')
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))

        TextInput({ placeholder: '请输入 API Key', text: this.apiKey })
          .fontSize(14)
          .type(InputType.Password)
          .onChange((value: string) => {
            this.apiKey = value;
          })
          .width('100%')
          .height(40)
          .backgroundColor($r('app.color.background_tertiary'))
          .borderRadius(8)
      }
      .width('100%')

      Row({ space: 12 }) {
        Button('保存')
          .fontSize(14)
          .height(36)
          .backgroundColor($r('app.color.primary'))
          .fontColor('#FFFFFF')
          .layoutWeight(1)
          .onClick(() => {
            this.saveSettings();
          })

        Button('取消')
          .fontSize(14)
          .height(36)
          .backgroundColor($r('app.color.background_tertiary'))
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .onClick(() => {
            this.showConfig = false;
          })
      }
      .width('100%')

      Text('API Key 获取方式:')
        .fontSize(11)
        .fontColor($r('app.color.text_tertiary'))
        .margin({ top: 8 })

      Text('国家超算互联网平台: chat.scnet.cn (选择带「接口服务」的模型)')
        .fontSize(10)
        .fontColor($r('app.color.text_tertiary'))
    }
    .width('100%')
    .padding(16)
    .alignItems(HorizontalAlign.Start)
    .layoutWeight(1)
  }

  /**
   * 构建聊天面板
   * 显示消息列表和输入区域
   */
  @Builder
  buildChatPanel() {
    Column() {
      if (this.messages.length === 0) {
        Column() {
          Text('📚 课程学习助手')
            .fontSize(14)
            .fontColor($r('app.color.text_primary'))
            .margin({ bottom: 8 })

          if (this.currentSemesterInfo) {
            Text(SemesterUtils.getSemesterDescription(this.enrollmentYear))
              .fontSize(12)
              .fontColor($r('app.color.primary'))
              .margin({ bottom: 8 })

            Text(`专业: ${this.major || '未获取'}`)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
              .margin({ bottom: 16 })
          }

          Text('你可以问我：')
            .fontSize(12)
            .fontColor($r('app.color.text_primary'))
            .margin({ bottom: 8 })

          Column({ space: 8 }) {
            Button('本学期有哪些课程？')
              .width('100%')
              .height(36)
              .fontSize(13)
              .backgroundColor($r('app.color.background_tertiary'))
              .fontColor($r('app.color.text_primary'))
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.inputText = '本学期有哪些课程？';
                this.sendMessage();
              })

            Button('考试课怎么备考？')
              .width('100%')
              .height(36)
              .fontSize(13)
              .backgroundColor($r('app.color.background_tertiary'))
              .fontColor($r('app.color.text_primary'))
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.inputText = '考试课怎么备考？';
                this.sendMessage();
              })

            Button('推荐自学什么技术？')
              .width('100%')
              .height(36)
              .fontSize(13)
              .backgroundColor($r('app.color.background_tertiary'))
              .fontColor($r('app.color.text_primary'))
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.inputText = '推荐自学什么技术？';
                this.sendMessage();
              })

            Button('生成完整学习计划')
              .width('100%')
              .height(36)
              .fontSize(13)
              .backgroundColor($r('app.color.primary'))
              .fontColor('#FFFFFF')
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.startPlanAnalysis();
              })
          }
          .width('100%')

          if (!this.apiKey) {
            Text('请先配置 API Key')
              .fontSize(12)
              .fontColor($r('app.color.warning'))
              .margin({ top: 12 })
          }
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
        .padding(16)
      } else {
        List({ scroller: this.scroller }) {
          ForEach(this.messages, (msg: ChatMessageItem) => {
            ListItem() {
              Row() {
                if (msg.role === 'user') {
                  Blank()
                  Text(msg.content)
                    .fontSize(14)
                    .fontColor('#FFFFFF')
                    .padding(12)
                    .backgroundColor($r('app.color.primary'))
                    .borderRadius({ topLeft: 16, topRight: 4, bottomLeft: 16, bottomRight: 16 })
                    .constraintSize({ maxWidth: '75%' })
                } else {
                  Text(msg.content)
                    .fontSize(14)
                    .fontColor($r('app.color.text_primary'))
                    .padding(12)
                    .backgroundColor($r('app.color.background_tertiary'))
                    .borderRadius({ topLeft: 4, topRight: 16, bottomLeft: 16, bottomRight: 16 })
                    .constraintSize({ maxWidth: '75%' })
                  Blank()
                }
              }
              .width('100%')
              .margin({ top: 8, bottom: 8 })
            }
          })
          
          if (this.isLoading) {
            ListItem() {
              Row() {
                LoadingProgress()
                  .width(20)
                  .height(20)
                  .color($r('app.color.primary'))
                Text('AI 正在思考中...')
                  .fontSize(12)
                  .fontColor($r('app.color.text_secondary'))
                  .margin({ left: 8 })
              }
              .padding(12)
              .backgroundColor($r('app.color.background_tertiary'))
              .borderRadius({ topLeft: 4, topRight: 16, bottomLeft: 16, bottomRight: 16 })
            }
            .margin({ top: 8, bottom: 8 })
          }
        }
        .width('100%')
        .layoutWeight(1)
        .padding({ left: 16, right: 16 })
        .scrollBar(BarState.Auto)
      }

      Row() {
        TextInput({ placeholder: '输入消息...', text: this.inputText })
          .layoutWeight(1)
          .height(40)
          .backgroundColor($r('app.color.background_tertiary'))
          .borderRadius(20)
          .padding({ left: 16, right: 16 })
          .enabled(!this.isLoading)
          .onChange((value: string) => {
            this.inputText = value;
          })
          .onSubmit(() => {
            this.sendMessage();
          })

        Button({ type: ButtonType.Circle }) {
          if (this.isLoading) {
            LoadingProgress()
              .width(20)
              .height(20)
              .color('#FFFFFF')
          } else {
            Image($r('app.media.paperplane'))
              .width(20)
              .height(20)
              .fillColor('#FFFFFF')
          }
        }
        .width(40)
        .height(40)
        .backgroundColor(this.inputText.trim() && !this.isLoading ? 
          $r('app.color.primary') : $r('app.color.text_tertiary'))
        .enabled(!!this.inputText.trim() && !this.isLoading)
        .onClick(() => {
          this.sendMessage();
        })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 8, bottom: 16 })
    }
    .width('100%')
    .layoutWeight(1)
  }
}
