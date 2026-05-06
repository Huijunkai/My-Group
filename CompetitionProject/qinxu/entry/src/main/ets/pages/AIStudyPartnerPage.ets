/**
 * AI学习伙伴页面模块
 * 
 * 提供AI驱动的学习辅助功能，包括：
 * - 学习方向规划（考研/考公/就业/出国）
 * - AI问答对话
 * - 知识测验
 * - 聊天历史管理
 * 
 * @module AIStudyPartnerPage
 */

import { router } from '@kit.ArkUI';
import { DataStore, Grade } from '../utils/DataStore';
import { AIService, AIResponse, AIModel, getModelDisplayName, getAllModels, GradeData, TrainingCourseData } from '../utils/AIService';
import { SemesterUtils, SemesterInfo } from '../utils/SemesterUtils';
import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

/**
 * 聊天消息项接口
 */
interface ChatMessageItem {
  /** 消息角色 */
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

/**
 * 问题项接口
 */
interface QuestionItem {
  /** 问题内容 */
  question: string;
  /** 选项列表 */
  options: string[];
  /** 正确答案索引 */
  correctAnswer: number;
  /** 答案解析 */
  explanation: string;
}

/** 学习方向类型 */
type Direction = 'postgraduate' | 'civil_service' | 'employment' | 'abroad';

/**
 * 方向选项接口
 */
interface DirectionOption {
  /** 方向键值 */
  key: Direction;
  /** 显示标签 */
  label: string;
  /** 图标 */
  icon: string;
  /** 描述说明 */
  description: string;
}

/** 偏好设置存储名称 */
const PREFERENCES_NAME = 'ai_study_partner_preferences';
/** AI API Key存储键名 */
const KEY_AI_API_KEY = 'ai_api_key';
/** AI模型存储键名 */
const KEY_AI_MODEL = 'ai_model';
/** 聊天历史存储键名 */
const KEY_CHAT_HISTORIES = 'chat_histories';

/** 学习方向选项列表 */
const DIRECTION_OPTIONS: DirectionOption[] = [
  { key: 'postgraduate', label: '考研', icon: '📚', description: '研究生入学考试' },
  { key: 'civil_service', label: '考公', icon: '🏛️', description: '公务员/事业单位' },
  { key: 'employment', label: '就业', icon: '💼', description: '企业求职就业' },
  { key: 'abroad', label: '出国', icon: '✈️', description: '海外留学深造' }
];

@Entry
@Component
struct AIStudyPartnerPage {
  @State messages: ChatMessageItem[] = [];
  @State inputText: string = '';
  @State isLoading: boolean = false;
  @State showConfig: boolean = false;
  @State showHistory: boolean = false;
  @State showDirectionSelector: boolean = false;
  @State selectedDirection: Direction = 'postgraduate';
  @State chatHistories: ChatHistory[] = [];
  @State currentChatId: string = '';
  @StorageLink('ai_api_key') apiKey: string = '';
  @StorageLink('ai_model') selectedModel: AIModel = 'MiniMax-M2.5';
  @State grades: GradeData[] = [];
  @State courses: TrainingCourseData[] = [];
  @State major: string = '';
  @State enrollmentYear: string = '';
  @State currentSemesterInfo: SemesterInfo | null = null;
  @State questions: QuestionItem[] = [];
  @State showQuiz: boolean = false;
  @State currentQuestionIndex: number = 0;
  @State selectedAnswers: number[] = [];
  @State showAnswer: boolean = false;
  @State score: number = 0;
  private aiService: AIService = AIService.getInstance();
  private scroller: Scroller = new Scroller();
  private context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
  private dataStore: DataStore = DataStore.getInstance();

  async aboutToAppear() {
    this.loadSettings();
    this.loadChatHistories();
    await this.loadData();
  }

  private async loadData(): Promise<void> {
    const rawGrades = await this.dataStore.loadGrades();
    const gradeItems: GradeData[] = [];
    for (let i = 0; i < rawGrades.length; i++) {
      const g = rawGrades[i];
      const scoreNum = parseFloat(g.score);
      if (!isNaN(scoreNum)) {
        gradeItems.push({
          courseName: g.courseName,
          credit: parseFloat(g.credit) || 0,
          grade: g.score,
          score: scoreNum,
          semester: g.semester || '未知学期'
        });
      }
    }
    this.grades = gradeItems;

    const rawPlans = await this.dataStore.loadPlans();
    const courseItems: TrainingCourseData[] = [];
    for (let i = 0; i < rawPlans.length; i++) {
      const p = rawPlans[i];
      courseItems.push({
        courseName: p.courseName,
        courseCode: p.courseCode,
        credit: Number(p.credit || 0),
        totalHours: Number(p.totalHours || 0),
        courseType: p.courseType || '',
        courseAttribute: p.courseAttribute || '',
        examType: p.examType || '不考试',
        semester: p.semester || '未知学期'
      });
    }
    this.courses = courseItems;

    const studentInfo = await this.dataStore.loadStudentInfo();
    if (studentInfo) {
      this.major = studentInfo.major || '';
      this.enrollmentYear = studentInfo.enrollmentYear || '';
    }

    if (!this.enrollmentYear && gradeItems.length > 0) {
      this.enrollmentYear = SemesterUtils.inferEnrollmentYearFromGrades(gradeItems) || '';
    }

    this.currentSemesterInfo = SemesterUtils.getCurrentSemester();
  }

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

  private async saveChatHistories(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      await dataPreferences.put(KEY_CHAT_HISTORIES, JSON.stringify(this.chatHistories));
      await dataPreferences.flush();
    } catch (error) {
      console.error('保存历史记录失败:', JSON.stringify(error));
    }
  }

  private generateChatId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
  }

  private getChatTitle(): string {
    if (this.messages.length > 0) {
      const firstUserMsg = this.messages.find((m: ChatMessageItem) => m.role === 'user');
      if (firstUserMsg) {
        return firstUserMsg.content.substring(0, 20) + (firstUserMsg.content.length > 20 ? '...' : '');
      }
    }
    return '新对话';
  }

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

  private async newChat(): Promise<void> {
    await this.saveCurrentChat();
    this.currentChatId = '';
    this.messages = [];
    this.aiService.clearHistory();
    this.showHistory = false;
  }

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

  private async startAnalysis(direction: Direction): Promise<void> {
    this.isLoading = true;
    this.showDirectionSelector = false;
    
    const directionLabel = DIRECTION_OPTIONS.find(d => d.key === direction)?.label || '综合';
    const userMsg: ChatMessageItem = { role: 'user', content: `请为我制定【${directionLabel}】方向的学习规划` };
    this.messages.push(userMsg);
    
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
    }, 50);

    try {
      const response: AIResponse = await this.aiService.analyzeStudyPartner(
        this.grades,
        this.courses,
        this.major,
        this.enrollmentYear,
        this.currentSemesterInfo,
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
      const errorMsg: ChatMessageItem = { role: 'assistant', content: `请求失败: ${JSON.stringify(error)}` };
      this.messages.push(errorMsg);
    } finally {
      this.isLoading = false;
    }
  }

  private async generateQuiz(): Promise<void> {
    if (!this.apiKey) {
      this.showConfig = true;
      return;
    }

    this.isLoading = true;
    const userMsg: ChatMessageItem = { role: 'user', content: '生成专业题库测试' };
    this.messages.push(userMsg);
    
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
    }, 50);

    try {
      const response: AIResponse = await this.aiService.generateQuiz(
        this.grades,
        this.courses,
        this.major,
        this.selectedDirection
      );
      
      if (response.success) {
        console.info('AI返回内容:', response.content);
        this.questions = this.parseQuestions(response.content);
        
        if (this.questions.length > 0) {
          this.currentQuestionIndex = 0;
          this.selectedAnswers = [];
          this.showAnswer = false;
          this.score = 0;
          this.showQuiz = true;
          const assistantMsg: ChatMessageItem = { role: 'assistant', content: `已生成 ${this.questions.length} 道题目，开始测试！` };
          this.messages.push(assistantMsg);
        } else {
          const assistantMsg: ChatMessageItem = { role: 'assistant', content: `题库生成失败，AI返回格式不正确。请稍后重试或更换模型。` };
          this.messages.push(assistantMsg);
          console.error('解析题目失败，原始内容:', response.content);
        }
      } else {
        const errorMsg: ChatMessageItem = { role: 'assistant', content: `生成失败: ${response.error}` };
        this.messages.push(errorMsg);
      }
      await this.saveCurrentChat();
      
      setTimeout(() => {
        this.scroller.scrollEdge(Edge.Bottom);
      }, 50);
    } catch (error) {
      console.error('生成题库异常:', JSON.stringify(error));
      const errorMsg: ChatMessageItem = { role: 'assistant', content: `请求失败: ${JSON.stringify(error)}` };
      this.messages.push(errorMsg);
    } finally {
      this.isLoading = false;
    }
  }

  private parseQuestions(content: string): QuestionItem[] {
    const questions: QuestionItem[] = [];
    try {
      let jsonStr = content;
      
      const jsonMatch = content.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        jsonStr = jsonMatch[0];
      }
      
      const parsed = JSON.parse(jsonStr) as QuestionItem[];
      for (let i = 0; i < parsed.length; i++) {
        const q = parsed[i];
        if (q.question && q.options && q.correctAnswer !== undefined) {
          questions.push({
            question: q.question,
            options: q.options,
            correctAnswer: typeof q.correctAnswer === 'number' ? q.correctAnswer : parseInt(String(q.correctAnswer)),
            explanation: q.explanation || ''
          });
        }
      }
    } catch (error) {
      console.error('解析题目失败:', JSON.stringify(error));
      console.error('原始内容:', content);
    }
    return questions;
  }

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
      const response: AIResponse = await this.aiService.chat(userMessage);
      
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

  private selectAnswer(index: number): void {
    if (this.showAnswer) return;
    this.selectedAnswers[this.currentQuestionIndex] = index;
  }

  private nextQuestion(): void {
    if (this.currentQuestionIndex < this.questions.length - 1) {
      this.currentQuestionIndex++;
      this.showAnswer = false;
    } else {
      this.calculateScore();
    }
  }

  private calculateScore(): void {
    let correct = 0;
    for (let i = 0; i < this.questions.length; i++) {
      if (this.selectedAnswers[i] === this.questions[i].correctAnswer) {
        correct++;
      }
    }
    this.score = Math.round((correct / this.questions.length) * 100);
    this.showAnswer = true;
  }

  private closeQuiz(): void {
    this.showQuiz = false;
    this.questions = [];
    this.currentQuestionIndex = 0;
    this.selectedAnswers = [];
    this.showAnswer = false;
    this.score = 0;
  }

  build() {
    Column() {
      Column()
        .width('100%')
        .height(44)

      Row() {
        Button({ type: ButtonType.Circle }) {
          Image($r('app.media.chevron_left'))
            .width(24)
            .height(24)
            .fillColor($r('app.color.text_primary'))
        }
        .width(40)
        .height(40)
        .backgroundColor('transparent')
        .onClick(() => {
          router.back();
        })

        Text('AI学习伙伴')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))
          .layoutWeight(1)
          .textAlign(TextAlign.Center)

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
      .padding({ left: 8, right: 8, top: 8, bottom: 8 })
      .backgroundColor($r('app.color.background_primary'))

      if (this.showHistory) {
        this.buildHistoryPanel()
      } else if (this.showConfig) {
        this.buildConfigPanel()
      } else if (this.showQuiz) {
        this.buildQuizPanel()
      } else {
        this.buildChatPanel()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_primary'))
    .expandSafeArea([SafeAreaType.SYSTEM], [SafeAreaEdge.TOP])
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
          }, (history: ChatHistory) => history.id.toString())
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
          }, (model: AIModel) => model)
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

  @Builder
  buildChatPanel() {
    Column() {
      if (this.messages.length === 0) {
        Column() {
          Text('🎓 AI学习伙伴')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary'))
            .margin({ bottom: 8 })

          if (this.currentSemesterInfo) {
            Text(SemesterUtils.getSemesterDescription(this.enrollmentYear))
              .fontSize(12)
              .fontColor($r('app.color.primary'))
              .margin({ bottom: 4 })

            Text(`专业: ${this.major || '未获取'}`)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
              .margin({ bottom: 16 })
          }

          Text('选择你的毕业方向')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.text_primary'))
            .margin({ bottom: 12 })

          Column({ space: 8 }) {
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
              .onClick(() => {
                this.selectedDirection = option.key;
                this.startAnalysis(option.key);
              })
            }, (option: DirectionOption) => option.key)
          }
          .width('100%')

          if (this.grades.length > 0) {
            Button('生成题库测试')
              .margin({ top: 16 })
              .backgroundColor($r('app.color.info'))
              .fontColor('#FFFFFF')
              .enabled(!this.isLoading && this.apiKey !== '')
              .onClick(() => {
                this.generateQuiz();
              })
          }

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
          ForEach(this.messages, (msg: ChatMessageItem, index: number) => {
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
          }, (msg: ChatMessageItem, index: number) => `msg_${index}_${msg.role}`)
          
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

        Row({ space: 8 }) {
          Button('题库')
            .fontSize(12)
            .height(36)
            .backgroundColor($r('app.color.info'))
            .fontColor('#FFFFFF')
            .enabled(!this.isLoading && this.apiKey !== '')
            .onClick(() => {
              this.generateQuiz();
            })

          Button('方向')
            .fontSize(12)
            .height(36)
            .backgroundColor($r('app.color.background_tertiary'))
            .fontColor($r('app.color.text_primary'))
            .onClick(() => {
              this.showDirectionSelector = true;
            })
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 4, bottom: 4 })
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
    .bindSheet($$this.showDirectionSelector, this.buildDirectionSheet(), {
      height: 400,
      backgroundColor: $r('app.color.background_primary'),
      dragBar: true
    })
  }

  @Builder
  buildDirectionSheet() {
    Column() {
      Text('选择毕业方向')
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
        .backgroundColor(this.selectedDirection === option.key ? 
          $r('app.color.info_light') : $r('app.color.background_tertiary'))
        .borderRadius(12)
        .margin({ bottom: 8 })
        .onClick(() => {
          this.selectedDirection = option.key;
          this.showDirectionSelector = false;
          this.startAnalysis(option.key);
        })
      }, (option: DirectionOption) => option.key)
    }
    .width('100%')
    .padding(20)
  }

  @Builder
  buildQuizPanel() {
    Column() {
      Row() {
        Text(`题目 ${this.currentQuestionIndex + 1}/${this.questions.length}`)
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))

        Blank()

        Button('退出')
          .fontSize(12)
          .height(28)
          .backgroundColor($r('app.color.background_tertiary'))
          .fontColor($r('app.color.text_primary'))
          .onClick(() => {
            this.closeQuiz();
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      if (this.questions.length > 0 && this.currentQuestionIndex < this.questions.length) {
        Column() {
          Text(this.questions[this.currentQuestionIndex].question)
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.text_primary'))
            .width('100%')
            .margin({ bottom: 20 })

          ForEach(this.questions[this.currentQuestionIndex].options, (option: string, index: number) => {
            Row() {
              Text(`${String.fromCharCode(65 + index)}. ${option}`)
                .fontSize(14)
                .fontColor(this.showAnswer && index === this.questions[this.currentQuestionIndex].correctAnswer ? 
                  '#52c41a' : $r('app.color.text_primary'))
            }
            .width('100%')
            .padding(14)
            .backgroundColor(
              this.showAnswer && index === this.questions[this.currentQuestionIndex].correctAnswer ? 
                'rgba(82, 196, 26, 0.1)' :
                (this.selectedAnswers[this.currentQuestionIndex] === index ? 
                  $r('app.color.info_light') : $r('app.color.background_tertiary'))
            )
            .borderRadius(10)
            .margin({ bottom: 8 })
            .onClick(() => {
              this.selectAnswer(index);
            })
          }, (option: string, index: number) => `q${this.currentQuestionIndex}_opt${index}`)

          if (this.showAnswer && this.questions[this.currentQuestionIndex].explanation) {
            Text(`解析: ${this.questions[this.currentQuestionIndex].explanation}`)
              .fontSize(12)
              .fontColor($r('app.color.text_secondary'))
              .width('100%')
              .padding(12)
              .backgroundColor($r('app.color.background_tertiary'))
              .borderRadius(8)
              .margin({ top: 8 })
          }

          if (this.showAnswer) {
            if (this.currentQuestionIndex < this.questions.length - 1) {
              Button('下一题')
                .width('100%')
                .height(44)
                .margin({ top: 20 })
                .backgroundColor($r('app.color.primary'))
                .fontColor('#FFFFFF')
                .onClick(() => {
                  this.nextQuestion();
                })
            } else {
              Column() {
                Text(`测试完成！得分: ${this.score}分`)
                  .fontSize(18)
                  .fontWeight(FontWeight.Bold)
                  .fontColor(this.score >= 60 ? '#52c41a' : '#f5222d')
                  .margin({ bottom: 16 })

                Button('完成')
                  .width('100%')
                  .height(44)
                  .backgroundColor($r('app.color.primary'))
                  .fontColor('#FFFFFF')
                  .onClick(() => {
                    this.closeQuiz();
                  })
              }
              .margin({ top: 20 })
            }
          } else if (this.selectedAnswers[this.currentQuestionIndex] !== undefined) {
            Button('确认答案')
              .width('100%')
              .height(44)
              .margin({ top: 20 })
              .backgroundColor($r('app.color.primary'))
              .fontColor('#FFFFFF')
              .onClick(() => {
                this.showAnswer = true;
              })
          }
        }
        .width('100%')
        .padding(16)
        .layoutWeight(1)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background_primary'))
  }
}

export { AIStudyPartnerPage };
