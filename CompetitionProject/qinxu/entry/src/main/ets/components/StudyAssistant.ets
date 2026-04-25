/**
 * AI学习助手组件模块
 * 
 * 提供AI驱动的学业规划功能，包括：
 * - 成绩分析与建议
 * - 学习规划指导
 * - 多种发展方向分析
 * - 聊天历史管理
 * 
 * @module StudyAssistant
 */

import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { pasteboard } from '@kit.BasicServicesKit';
import { AIService, GradeData, AIResponse, AIModel, getModelDisplayName, getAllModels } from '../utils/AIService';
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

/** 发展方向类型 */
export type DevelopmentDirection = 'general' | 'civil_service' | 'certification' | 'employment' | 'abroad';

/**
 * 发展方向选项接口
 */
interface DirectionOption {
  /** 方向键值 */
  key: DevelopmentDirection;
  /** 显示标签 */
  label: string;
  /** 图标 */
  icon: string;
  /** 描述说明 */
  description: string;
}

/** 发展方向选项列表 */
const DIRECTION_OPTIONS: DirectionOption[] = [
  { key: 'general', label: '综合分析', icon: '📊', description: '全面分析学业情况' },
  { key: 'civil_service', label: '考公指导', icon: '🏛️', description: '公务员考试规划' },
  { key: 'certification', label: '考证规划', icon: '📜', description: '职业资格证书备考' },
  { key: 'employment', label: '就业指导', icon: '💼', description: '求职与职业发展' },
  { key: 'abroad', label: '出国留学', icon: '✈️', description: '留学申请规划' }
];

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'theme_preferences';
/** AI API Key存储键名 */
const KEY_AI_API_KEY = 'ai_api_key';
/** AI模型存储键名 */
const KEY_AI_MODEL = 'ai_model';
/** 聊天历史存储键名 */
const KEY_CHAT_HISTORIES = 'chat_histories';

/**
 * AI学习助手组件
 * 
 * 提供AI驱动的学业规划和成绩分析功能
 * 
 * @example
 * ```typescript
 * StudyAssistant({
 *   grades: gradeList,
 *   enrollmentYear: '2022',
 *   currentSemesterInfo: semesterInfo
 * })
 * ```
 */
@Entry
@Component
export struct StudyAssistant {
  /** 成绩数据 */
  @Prop grades: GradeData[] = [];
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
  /** 显示发展方向选择器标志 */
  @State showDirectionSelector: boolean = false;
  /** 选中的发展方向 */
  @State selectedDirection: DevelopmentDirection = 'general';
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
    this.loadConfig();
    this.loadChatHistories();
  }

  /**
   * 加载配置
   * 从全局存储加载API配置
   */
  private loadConfig(): void {
    if (this.apiKey) {
      this.aiService.setConfig({
        apiKey: this.apiKey,
        model: this.selectedModel
      });
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
      console.error('StudyAssistant: Failed to load chat histories:', JSON.stringify(error));
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
      console.error('StudyAssistant: Failed to save chat histories:', JSON.stringify(error));
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
   * 保存配置
   * 将API配置持久化到本地存储
   */
  private async saveConfig(): Promise<void> {
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
      console.error('StudyAssistant: Failed to save config:', JSON.stringify(error));
    }
  }

  /**
   * 判断是否为成绩相关查询
   * 
   * @param message - 用户消息
   * @returns 是否包含成绩相关关键词
   */
  private isGradeRelatedQuery(message: string): boolean {
    const keywords: string[] = [
      '成绩', '分数', '绩点', 'gpa', '学分', '课程',
      '分析', '学习', '建议', '规划', '考研',
      '提高', '提升', '薄弱', '优势', '专业'
    ];
    const lowerMessage = message.toLowerCase();
    for (let i = 0; i < keywords.length; i++) {
      if (lowerMessage.includes(keywords[i])) {
        return true;
      }
    }
    return false;
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
      let response: AIResponse;
      
      if (this.grades.length > 0 && this.isGradeRelatedQuery(userMessage)) {
        response = await this.aiService.analyzeGrades(
          this.grades,
          this.enrollmentYear,
          this.currentSemesterInfo,
          userMessage
        );
      } else {
        response = await this.aiService.chat(userMessage);
      }
      
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

  /**
   * 开始成绩分析
   * 根据选定的发展方向分析成绩
   * 
   * @param direction - 发展方向
   */
  private async startGradeAnalysis(direction: DevelopmentDirection = 'general'): Promise<void> {
    if (this.grades.length === 0) {
      const errorMsg: ChatMessageItem = { role: 'assistant', content: '暂无成绩数据可供分析。' };
      this.messages.push(errorMsg);
      return;
    }

    this.isLoading = true;
    this.showDirectionSelector = false;
    
    const directionLabel = DIRECTION_OPTIONS.find(d => d.key === direction)?.label || '综合分析';
    const userMsg: ChatMessageItem = { role: 'user', content: `请针对【${directionLabel}】方向分析我的成绩` };
    this.messages.push(userMsg);
    
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
    }, 50);

    try {
      const response: AIResponse = await this.aiService.analyzeGrades(
        this.grades,
        this.enrollmentYear,
        this.currentSemesterInfo,
        undefined,
        direction
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
   * 清空对话
   * 清除当前对话的所有消息
   */
  private clearConversation(): void {
    this.messages = [];
    this.aiService.clearHistory();
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

        Text('AI学习助手')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        Blank()

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
    .backgroundColor($r('app.color.background_primary'))
    .borderRadius({ topLeft: 16, topRight: 16 })
  }

  @Builder
  buildHistoryPanel() {
    Column({ space: 12 }) {
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
    .alignItems(HorizontalAlign.Start)
  }

  private formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
      return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    } else if (days === 1) {
      return '昨天';
    } else if (days < 7) {
      return `${days}天前`;
    } else {
      return `${date.getMonth() + 1}/${date.getDate()}`;
    }
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

      TextInput({ placeholder: '请输入 API Key', text: this.apiKey })
        .width('100%')
        .height(44)
        .type(InputType.Password)
        .backgroundColor($r('app.color.background_tertiary'))
        .borderRadius(8)
        .onChange((value: string) => {
          this.apiKey = value;
        })

      Row({ space: 8 }) {
        Button('取消')
          .layoutWeight(1)
          .height(40)
          .backgroundColor($r('app.color.background_tertiary'))
          .fontColor($r('app.color.text_primary'))
          .onClick(() => {
            this.showConfig = false;
          })

        Button('保存')
          .layoutWeight(1)
          .height(40)
          .backgroundColor($r('app.color.primary'))
          .fontColor('#FFFFFF')
          .onClick(async () => {
            await this.saveConfig();
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
  }

  @Builder
  buildChatPanel() {
    Column() {
      if (this.messages.length === 0) {
        Column() {
          Text('👋 你好！我是你的学业规划助手')
            .fontSize(14)
            .fontColor($r('app.color.text_primary'))
            .margin({ bottom: 8 })

          if (this.currentSemesterInfo) {
            Text(SemesterUtils.getSemesterDescription(this.enrollmentYear))
              .fontSize(12)
              .fontColor($r('app.color.primary'))
              .margin({ bottom: 8 })
          }

          Text('我可以帮你：\n• 分析成绩和绩点(GPA)\n• 制定学习提升计划\n• 考研/考公/就业规划\n• 解答学习相关问题')
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .textAlign(TextAlign.Center)

          if (this.grades.length > 0) {
            Button('分析我的成绩')
              .margin({ top: 16 })
              .backgroundColor($r('app.color.primary'))
              .fontColor('#FFFFFF')
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.showDirectionSelector = true;
              })
          }

          if (!this.apiKey) {
            Text('请先配置 API Key')
              .fontSize(12)
              .fontColor($r('app.color.warning'))
              .margin({ top: 12 })
          }

          Text('考研信息: yz.chsi.com.cn')
            .fontSize(11)
            .fontColor($r('app.color.primary'))
            .margin({ top: 16 })
            .decoration({ type: TextDecorationType.Underline })
            .onClick(() => {
              this.copyToClipboard('https://yz.chsi.com.cn/');
            })
        }
        .width('100%')
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
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
        .backgroundColor($r('app.color.primary'))
        .margin({ left: 8 })
        .enabled(!this.isLoading && this.inputText.trim() !== '')
        .onClick(() => {
          this.sendMessage();
        })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 8, bottom: 16 })
    }
    .width('100%')
    .layoutWeight(1)
    
    if (this.showDirectionSelector) {
      this.buildDirectionSelector()
    }
  }

  @Builder
  buildDirectionSelector() {
    Column() {
      Column() {
        Text('选择发展方向')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .margin({ bottom: 16 })

        ForEach(DIRECTION_OPTIONS, (option: DirectionOption) => {
          Row() {
            Text(option.icon)
              .fontSize(24)
              .margin({ right: 12 })

            Column() {
              Text(option.label)
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor($r('app.color.text_primary'))

              Text(option.description)
                .fontSize(11)
                .fontColor($r('app.color.text_secondary'))
                .margin({ top: 2 })
            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1)
          }
          .width('100%')
          .padding(12)
          .backgroundColor($r('app.color.background_tertiary'))
          .borderRadius(12)
          .margin({ bottom: 8 })
          .onClick(() => {
            this.selectedDirection = option.key;
            this.startGradeAnalysis(option.key);
          })
        })

        Button('取消')
          .width('100%')
          .height(40)
          .backgroundColor($r('app.color.background_tertiary'))
          .fontColor($r('app.color.text_primary'))
          .margin({ top: 8 })
          .onClick(() => {
            this.showDirectionSelector = false;
          })
      }
      .width('85%')
      .padding(20)
      .backgroundColor($r('app.color.background_primary'))
      .borderRadius(20)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor('rgba(0, 0, 0, 0.5)')
    .onClick(() => {
      this.showDirectionSelector = false;
    })
  }
}
