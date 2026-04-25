/**
 * AI服务模块
 * 
 * 提供AI对话和智能分析功能，包括：
 * - 多模型支持（MiniMax、Qwen、DeepSeek）
 * - 对话历史管理
 * - 成绩分析
 * - 学习规划
 * 
 * @module AIService
 */

import http from '@ohos.net.http';
import { SemesterUtils, SemesterInfo, StudentSemesterInfo } from './SemesterUtils';

/**
 * 聊天消息接口
 */
export interface ChatMessage {
  /** 消息角色 */
  role: 'system' | 'user' | 'assistant';
  /** 消息内容 */
  content: string;
}

/**
 * AI响应接口
 */
export interface AIResponse {
  /** 是否成功 */
  success: boolean;
  /** 响应内容 */
  content: string;
  /** 错误信息 */
  error?: string;
}

/** AI模型类型 */
export type AIModel = 'MiniMax-M2.5' | 'Qwen3-235B-A22B' | 'DeepSeek-R1-7B' | 'DeepSeek-R1-32B';

/**
 * AI配置接口
 */
export interface AIConfig {
  /** API密钥 */
  apiKey: string;
  /** 模型名称 */
  model: AIModel;
  /** 代理URL */
  proxyUrl?: string;
}

/**
 * 成绩数据接口
 */
export interface GradeData {
  /** 课程名称 */
  courseName: string;
  /** 学分 */
  credit: number;
  /** 成绩等级 */
  grade: string;
  /** 分数 */
  score: number;
  /** 学期 */
  semester: string;
  /** 绩点 */
  gpa?: number;
}

/**
 * 消息项接口
 */
interface MessageItem {
  /** 角色 */
  role: string;
  /** 内容 */
  content: string;
}

/**
 * 代理请求接口
 */
interface ProxyRequest {
  /** 消息列表 */
  messages: MessageItem[];
  /** API密钥 */
  apiKey: string;
  /** 模型 */
  model: AIModel;
}

/**
 * 代理响应接口
 */
interface ProxyResponse {
  /** 是否成功 */
  success: boolean;
  /** 内容 */
  content: string;
  /** 消息 */
  message?: string;
  /** 详情 */
  details?: string;
  /** 状态码 */
  statusCode?: number;
}

/**
 * 消息输入接口
 */
interface MessageInput {
  /** 角色 */
  role: string;
  /** 内容 */
  content: string;
}

/** 模型显示名称映射 */
const MODEL_DISPLAY_NAMES: Record<AIModel, string> = {
  'MiniMax-M2.5': 'MiniMax (快)',
  'Qwen3-235B-A22B': 'Qwen3 (强)',
  'DeepSeek-R1-7B': 'DeepSeek R1-7B',
  'DeepSeek-R1-32B': 'DeepSeek R1-32B'
};

const MODEL_API_NAMES: Record<AIModel, string> = {
  'MiniMax-M2.5': 'MiniMax-M2.5',
  'Qwen3-235B-A22B': 'Qwen3-235B-A22B',
  'DeepSeek-R1-7B': 'DeepSeek-R1-Distill-Qwen-7B',
  'DeepSeek-R1-32B': 'DeepSeek-R1-Distill-Qwen-32B'
};

/**
 * 获取模型显示名称
 * 
 * @param model - AI模型
 * @returns 显示名称
 */
export function getModelDisplayName(model: AIModel): string {
  return MODEL_DISPLAY_NAMES[model] || model;
}

/**
 * 获取所有可用模型列表
 * 
 * @returns 模型列表
 */
export function getAllModels(): AIModel[] {
  return ['MiniMax-M2.5', 'Qwen3-235B-A22B', 'DeepSeek-R1-7B', 'DeepSeek-R1-32B'];
}

/**
 * AI服务类
 * 
 * 提供AI对话和智能分析功能
 * 使用单例模式
 * 
 * @example
 * ```typescript
 * const aiService = AIService.getInstance();
 * aiService.setConfig({ apiKey: 'your-key', model: 'MiniMax-M2.5' });
 * const response = await aiService.chat('你好');
 * ```
 */
export class AIService {
  /** 单例实例 */
  private static instance: AIService;
  /** AI配置 */
  private config: AIConfig | null = null;
  /** 对话历史 */
  private conversationHistory: ChatMessage[] = [];
  /** 代理服务URL */
  private readonly proxyUrl: string = 'http://124.70.92.199:3001/api/ai/chat';

  /**
   * 获取单例实例
   * 
   * @returns AIService实例
   */
  static getInstance(): AIService {
    if (!AIService.instance) {
      AIService.instance = new AIService();
    }
    return AIService.instance;
  }

  setConfig(config: AIConfig): void {
    this.config = config;
    this.conversationHistory = [];
  }

  setSystemPrompt(prompt: string): void {
    const systemMessage: ChatMessage = {
      role: 'system',
      content: prompt
    };
    this.conversationHistory = [systemMessage];
  }

  clearHistory(): void {
    const systemPrompt = this.conversationHistory.find((m: ChatMessage) => m.role === 'system');
    this.conversationHistory = systemPrompt ? [systemPrompt] : [];
  }

  setHistoryFromMessages(messages: MessageInput[]): void {
    this.conversationHistory = messages.map((m: MessageInput): ChatMessage => {
      const msg: ChatMessage = {
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.content
      };
      return msg;
    });
  }

  async chat(userMessage: string): Promise<AIResponse> {
    if (!this.config) {
      const response: AIResponse = {
        success: false,
        content: '',
        error: '请先配置 API Key'
      };
      return response;
    }

    const userMsg: ChatMessage = {
      role: 'user',
      content: userMessage
    };
    this.conversationHistory.push(userMsg);

    try {
      const response = await this.callProxyAPI();
      if (response.success && response.content) {
        const assistantMsg: ChatMessage = {
          role: 'assistant',
          content: response.content
        };
        this.conversationHistory.push(assistantMsg);
      }
      return response;
    } catch (error) {
      const errorObj = error as Record<string, Object>;
      const errorCode = errorObj['code'] as number;
      let errorMessage = '请求失败';
      
      if (errorCode === 2300006) {
        errorMessage = '无法连接到服务器，请检查网络';
      } else if (errorCode === 2300001) {
        errorMessage = '请求超时，请稍后重试';
      } else {
        errorMessage = `请求失败: ${JSON.stringify(error)}`;
      }
      
      const errorResponse: AIResponse = {
        success: false,
        content: '',
        error: errorMessage
      };
      return errorResponse;
    }
  }

  private async callProxyAPI(): Promise<AIResponse> {
    if (!this.config) {
      const response: AIResponse = { success: false, content: '', error: '未配置' };
      return response;
    }

    const httpRequest = http.createHttp();
    const url: string = this.config.proxyUrl || this.proxyUrl;

    console.info(`AIService: Using proxy: ${url}`);
    console.info(`AIService: API Key (first 10): ${this.config.apiKey ? this.config.apiKey.substring(0, 10) + '...' : 'EMPTY'}`);

    try {
      const messages: MessageItem[] = this.conversationHistory.map((m: ChatMessage): MessageItem => {
        const msg: MessageItem = {
          role: m.role,
          content: m.content
        };
        return msg;
      });

      const requestBody: ProxyRequest = {
        messages: messages,
        apiKey: this.config.apiKey,
        model: MODEL_API_NAMES[this.config.model] as AIModel
      };

      console.info(`AIService: Model: ${this.config.model} -> ${MODEL_API_NAMES[this.config.model]}`);

      console.info(`AIService: Sending request via proxy...`);

      const response = await httpRequest.request(url, {
        method: http.RequestMethod.POST,
        header: { 'Content-Type': 'application/json' },
        extraData: JSON.stringify(requestBody),
        connectTimeout: 120000,
        readTimeout: 120000
      });

      console.info(`AIService: Proxy response code: ${response.responseCode}`);

      if (response.responseCode === 200) {
        const result: ProxyResponse = JSON.parse(response.result as string) as ProxyResponse;
        if (result.success) {
          const successResponse: AIResponse = { success: true, content: result.content };
          return successResponse;
        } else {
          const errorMsg = result.message || result.details || 'AI 请求失败';
          if (errorMsg.toLowerCase().includes('insufficient balance') || 
              errorMsg.includes('余额不足') ||
              errorMsg.includes('额度不足')) {
            const errorResponse: AIResponse = {
              success: false,
              content: '',
              error: 'Token 额度不足，请前往国家超算互联网平台充值\nhttps://www.scnet.cn'
            };
            return errorResponse;
          }
          const errorResponse: AIResponse = {
            success: false,
            content: '',
            error: errorMsg
          };
          return errorResponse;
        }
      } else if (response.responseCode === 401) {
        const errorResponse: AIResponse = {
          success: false,
          content: '',
          error: 'Token 额度不足，请前往国家超算互联网平台充值\nhttps://www.scnet.cn'
        };
        return errorResponse;
      } else {
        let errorDetail = `代理服务器错误: ${response.responseCode}`;
        try {
          const errorResult = JSON.parse(response.result as string) as ProxyResponse;
          errorDetail = errorResult.message || errorDetail;
          if (errorDetail.toLowerCase().includes('insufficient balance') ||
              errorDetail.includes('余额不足') ||
              errorDetail.includes('额度不足')) {
            const errorResponse: AIResponse = {
              success: false,
              content: '',
              error: 'Token 额度不足，请前往国家超算互联网平台充值\nhttps://www.scnet.cn'
            };
            return errorResponse;
          }
        } catch (e) {
          // 无法解析
        }
        const errorResponse: AIResponse = {
          success: false,
          content: '',
          error: errorDetail
        };
        return errorResponse;
      }
    } catch (error) {
      console.error('AIService: Proxy request failed:', JSON.stringify(error));
      const errorObj = error as Record<string, Object>;
      const errorCode = errorObj['code'] as number;
      let errorMessage = '请求失败';
      
      if (errorCode === 2300006) {
        errorMessage = '无法连接到代理服务器，请检查服务器地址';
      } else if (errorCode === 2300001) {
        errorMessage = '请求超时，请稍后重试';
      } else {
        errorMessage = `请求失败: ${JSON.stringify(error)}`;
      }
      
      const errorResponse: AIResponse = {
        success: false,
        content: '',
        error: errorMessage
      };
      return errorResponse;
    } finally {
      httpRequest.destroy();
    }
  }

  async analyzeGrades(
    grades: GradeData[],
    enrollmentYear?: string,
    currentSemesterInfo?: SemesterInfo | null,
    userQuestion?: string,
    direction: string = 'general'
  ): Promise<AIResponse> {
    const directionPrompts: Record<string, string> = {
      'general': `你是一位资深的大学学业规划顾问，擅长分析成绩、提供学习建议和考研规划。

【回复格式要求】
- 不要使用任何Markdown格式（如**加粗**、#标题、-列表符号等）
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 用换行分隔不同要点
- 重要信息用「」或【】标注

请用专业、鼓励的语气，给出具体可操作的建议。`,

      'civil_service': `你是一位资深的公务员考试规划顾问，擅长根据学生成绩分析适合报考的岗位，提供备考建议。

【回复格式要求】
- 不要使用任何Markdown格式
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 重要信息用「」或【】标注

【分析重点】
1. 根据专业和成绩分析适合报考的岗位类型
2. 行测、申论备考建议和时间规划
3. 推荐报考的岗位层级（国考/省考/事业单位）
4. 专业优势发挥和劣势弥补方案
5. 面试准备建议

公务员考试信息：国家公务员局 https://www.scs.gov.cn/`,

      'certification': `你是一位资深的职业资格认证规划顾问，擅长根据学生专业和成绩推荐适合考取的职业资格证书。

【回复格式要求】
- 不要使用任何Markdown格式
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 重要信息用「」或【】标注

【分析重点】
1. 根据专业推荐核心证书和加分证书
2. 证书考试时间安排和备考周期
3. 证书含金量和就业帮助分析
4. 备考资料和学习方法推荐
5. 考证优先级排序建议`,

      'employment': `你是一位资深的大学生就业指导顾问，擅长分析学生优势、提供求职建议和职业规划。

【回复格式要求】
- 不要使用任何Markdown格式
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 重要信息用「」或【】标注

【分析重点】
1. 根据成绩和课程分析适合的就业方向
2. 简历制作和面试技巧建议
3. 实习机会和求职渠道推荐
4. 薪资期望和行业前景分析
5. 职业发展路径规划

招聘信息：应届生求职网 https://www.yingjiesheng.com/`,

      'abroad': `你是一位资深的留学申请规划顾问，擅长根据学生背景推荐适合的留学方案。

【回复格式要求】
- 不要使用任何Markdown格式
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 重要信息用「」或【】标注

【分析重点】
1. 根据GPA分析可申请的学校档次
2. 语言考试（雅思/托福）备考建议
3. 推荐信、文书准备要点
4. 申请时间线和材料清单
5. 奖学金申请建议

留学信息：教育部留学服务中心 https://www.cscse.edu.cn/`
    };

    const systemPrompt = directionPrompts[direction] || directionPrompts['general'];
    this.setSystemPrompt(systemPrompt);

    if (grades.length === 0) {
      return this.chat('我是一名大学生，请给我一些大学学习建议，包括如何提高绩点、平衡理论与实践、以及考研准备。');
    }

    let totalCredits: number = 0;
    let totalScore: number = 0;
    let totalGPA: number = 0;
    
    for (let i = 0; i < grades.length; i++) {
      totalCredits += grades[i].credit;
      totalScore += grades[i].score;
      totalGPA += grades[i].gpa || 0;
    }
    
    const avgScore: number = totalScore / grades.length;
    const avgGPA: number = totalGPA / grades.length;
    
    const semesterCount = SemesterUtils.getSemesterCount(grades);
    const latestSemester = SemesterUtils.getLatestSemesterFromGrades(grades);
    
    const gradesBySemester = new Map<string, GradeData[]>();
    for (let i = 0; i < grades.length; i++) {
      const g = grades[i];
      const sem = g.semester || '未知学期';
      if (!gradesBySemester.has(sem)) {
        gradesBySemester.set(sem, []);
      }
      gradesBySemester.get(sem)?.push(g);
    }
    
    let semesterSummary = '';
    const sortedSemesters = SemesterUtils.sort(Array.from(gradesBySemester.keys()));
    for (let i = 0; i < sortedSemesters.length; i++) {
      const sem = sortedSemesters[i];
      const semGrades = gradesBySemester.get(sem) || [];
      let semCredits = 0;
      let semTotalScore = 0;
      for (let j = 0; j < semGrades.length; j++) {
        semCredits += semGrades[j].credit;
        semTotalScore += semGrades[j].score;
      }
      const semAvgScore = semTotalScore / semGrades.length;
      semesterSummary += `\n${SemesterUtils.format(sem)}: ${semGrades.length}门课, ${semCredits}学分, 均分${semAvgScore.toFixed(1)}`;
    }

    const gradeList: string = grades.map((g: GradeData) => 
      `${g.courseName}(${g.credit}学分): ${g.grade}分${g.gpa !== undefined ? `, 绩点${g.gpa.toFixed(1)}` : ''}`
    ).join('\n');

    const highScoreCourses: GradeData[] = grades.filter((g: GradeData) => g.score >= 85);
    const lowScoreCourses: GradeData[] = grades.filter((g: GradeData) => g.score < 70);
    
    let studentStatusInfo = '';
    if (enrollmentYear && currentSemesterInfo) {
      const studentInfo = SemesterUtils.getStudentCurrentSemester(enrollmentYear);
      if (studentInfo) {
        const gradeName = SemesterUtils.getGradeName(studentInfo.semesterNumber);
        studentStatusInfo = `\n\n【学生当前状态】
入学年份: ${enrollmentYear}级
当前学期: ${currentSemesterInfo.displayName}
所在年级: ${gradeName}（第${studentInfo.semesterNumber}学期）`;
      }
    }
    
    if (!studentStatusInfo && latestSemester) {
      studentStatusInfo = `\n\n【最近学期】
${latestSemester.displayName}`;
    }

    const contextInfo: string = `【学生当前状态】${studentStatusInfo}

【成绩清单】
${gradeList}

【学期统计】
共${semesterCount}个学期
${semesterSummary}

【总体统计】
总学分: ${totalCredits}
平均分: ${avgScore.toFixed(1)}
平均绩点: ${avgGPA.toFixed(2)}
优秀课程(≥85分): ${highScoreCourses.length}门
待提升课程(<70分): ${lowScoreCourses.length}门`;

    let prompt: string;
    if (userQuestion && userQuestion.trim()) {
      prompt = `${contextInfo}

用户问题: ${userQuestion}

请根据用户的成绩数据，回答用户的问题，给出具体可操作的建议。`;
    } else {
      prompt = `请分析我的成绩：
${contextInfo}

请根据我当前的年级和学期，从以下方面给出针对性建议：
1. 当前阶段学业分析和绩点提升方法
2. 学习优势和薄弱科目分析
3. 理论和实践学习建议
4. ${this.getStudySuggestionFocus(enrollmentYear, currentSemesterInfo)}
5. 考研规划建议（提醒访问yz.chsi.com.cn查询最新信息）`;
    }

    return this.chat(prompt);
  }

  private getStudySuggestionFocus(enrollmentYear?: string, currentSemesterInfo?: SemesterInfo | null): string {
    if (!enrollmentYear || !currentSemesterInfo) {
      return '后续学期学习规划';
    }

    const studentInfo = SemesterUtils.getStudentCurrentSemester(enrollmentYear);
    if (!studentInfo) {
      return '后续学期学习规划';
    }

    const semesterNum = studentInfo.semesterNumber;

    if (semesterNum <= 2) {
      return '大一阶段基础课程学习建议和专业方向探索';
    } else if (semesterNum <= 4) {
      return '大二阶段专业课程深化和竞赛/证书准备';
    } else if (semesterNum <= 6) {
      return '大三阶段专业课冲刺和考研/就业准备';
    } else {
      return '大四阶段毕业设计和职业规划';
    }
  }

  async analyzeTrainingPlan(
    courses: TrainingCourseData[],
    major?: string,
    enrollmentYear?: string,
    currentSemesterInfo?: SemesterInfo | null,
    userQuestion?: string
  ): Promise<AIResponse> {
    const systemPrompt: string = `你是一位资深的大学课程学习规划顾问，擅长根据培养计划制定学习方案，结合专业就业动态提供指导。

【回复格式要求】
- 不要使用任何Markdown格式（如**加粗**、#标题、-列表符号等）
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 用换行分隔不同要点
- 重要信息用「」或【】标注

【交互原则】
- 根据用户的具体问题提供针对性回答
- 如果用户问的是具体问题，直接回答，不要生成完整学习计划
- 如果用户没有特定问题或明确要求学习计划，才生成完整学习计划
- 保持对话的连贯性和交互性
- 鼓励用户提出更多具体问题

【分析重点】
1. 根据当前学期课程，制定学习计划和时间安排
2. 分析必修课和选修课的重点难点
3. 考试课程和考查课程的备考策略
4. 结合专业就业动态，推荐自学内容和技术方向
5. 提供实用的学习资源和方法建议

请用专业、鼓励的语气，给出具体可操作的建议。`;

    this.setSystemPrompt(systemPrompt);

    if (courses.length === 0) {
      return this.chat('我是一名大学生，请给我一些课程学习建议。');
    }

    const currentSemester = currentSemesterInfo ? currentSemesterInfo.raw : '';
    const currentCourses: TrainingCourseData[] = [];
    
    for (let i = 0; i < courses.length; i++) {
      if (courses[i].semester === currentSemester) {
        currentCourses.push(courses[i]);
      }
    }

    let currentCourseList = '';
    let examCourses = 0;
    let requiredCourses = 0;
    let totalCredits = 0;
    
    for (let i = 0; i < currentCourses.length; i++) {
      const course = currentCourses[i];
      currentCourseList += `\n${course.courseName} (${course.credit}学分, ${course.totalHours}学时)`;
      currentCourseList += ` - ${course.courseAttribute || '未知属性'}`;
      currentCourseList += ` - ${course.examType}`;
      totalCredits += course.credit;
      if (course.examType === '考试') {
        examCourses++;
      }
      if (course.courseAttribute && course.courseAttribute.includes('必修')) {
        requiredCourses++;
      }
    }

    let studentStatusInfo = '';
    if (enrollmentYear && currentSemesterInfo) {
      const studentInfo = SemesterUtils.getStudentCurrentSemester(enrollmentYear);
      if (studentInfo) {
        const gradeName = SemesterUtils.getGradeName(studentInfo.semesterNumber);
        studentStatusInfo = `入学年份: ${enrollmentYear}级
当前学期: ${currentSemesterInfo.displayName}
所在年级: ${gradeName}（第${studentInfo.semesterNumber}学期）`;
      }
    } else if (currentSemesterInfo) {
      studentStatusInfo = `当前学期: ${currentSemesterInfo.displayName}`;
    }

    const contextInfo: string = `【学生当前状态】
专业: ${major || '未知'}
${studentStatusInfo}

【本学期课程】
共${currentCourses.length}门课程，${totalCredits}学分
考试课: ${examCourses}门，必修课: ${requiredCourses}门
${currentCourseList}`;

    let prompt: string;
    
    if (userQuestion && userQuestion.trim()) {
      const question = userQuestion.trim().toLowerCase();
      const isPlanRequest = question.includes('学习计划') || 
                           question.includes('制定计划') || 
                           question.includes('规划') ||
                           question.includes('安排') ||
                           question === '生成本学期学习计划';
      
      if (isPlanRequest) {
        prompt = `${contextInfo}

用户请求: ${userQuestion}

请根据以上信息，为用户制定详细的本学期学习计划，包括：
1. 课程学习优先级和时间安排
2. 考试课程的备考重点
3. 必修课和选修课的学习策略
4. 推荐的学习资源和方法`;
      } else {
        prompt = `${contextInfo}

用户问题: ${userQuestion}

请根据用户的课程情况，直接回答用户的问题。保持回答的针对性和实用性。`;
      }
    } else {
      prompt = `${contextInfo}

请简要介绍本学期的课程情况，并询问用户想了解哪方面的内容，比如：
- 某门课程的学习方法
- 考试备考策略
- 就业方向建议
- 技术学习推荐等`;
    }

    return this.chat(prompt);
  }

  async analyzeStudyPartner(
    grades: GradeData[],
    courses: TrainingCourseData[],
    major?: string,
    enrollmentYear?: string,
    currentSemesterInfo?: SemesterInfo | null,
    direction: string = 'postgraduate'
  ): Promise<AIResponse> {
    const directionPrompts: Record<string, string> = {
      'postgraduate': `你是一位资深的研究生入学考试规划顾问，擅长根据学生成绩和课程背景制定考研规划。

【分析重点】
1. 根据专业和成绩分析适合报考的院校和专业方向
2. 考研科目（政治、英语、数学、专业课）备考规划
3. 推荐复习资料、教材和网课资源
4. 历年真题分析和重点知识点梳理
5. 复习时间轴和阶段性目标

考研信息：中国研究生招生信息网 https://yz.chsi.com.cn/`,

      'civil_service': `你是一位资深的公务员考试规划顾问，擅长根据学生背景制定公考规划。

【分析重点】
1. 根据专业分析适合报考的岗位类型（国考/省考/事业单位）
2. 行测、申论备考策略和时间安排
3. 推荐复习资料和刷题资源
4. 历年真题分析和高频考点
5. 面试技巧和注意事项

公考信息：国家公务员局 https://www.scs.gov.cn/`,

      'employment': `你是一位资深的大学生就业指导顾问，擅长根据学生背景制定求职规划。

【分析重点】
1. 根据专业和成绩分析适合的就业方向和岗位
2. 简历制作要点和面试技巧
3. 推荐需要掌握的技能和证书
4. 招聘信息获取渠道（BOSS直聘、智联招聘等）
5. 职业发展路径规划

招聘平台：BOSS直聘 https://www.zhipin.com/`,

      'abroad': `你是一位资深的留学申请规划顾问，擅长根据学生背景制定留学方案。

【分析重点】
1. 根据GPA分析可申请的学校档次
2. 语言考试（雅思/托福）备考规划
3. 推荐信、文书准备要点
4. 申请时间线和材料清单
5. 奖学金申请建议

留学信息：教育部留学服务中心 https://www.cscse.edu.cn/`
    };

    const systemPrompt = `你是一位专业的大学生学业规划顾问。

【回复格式要求】
- 不要使用任何Markdown格式
- 使用纯文本，用数字编号组织内容
- 每段内容简短，适合手机阅读
- 重要信息用「」或【】标注

${directionPrompts[direction] || directionPrompts['postgraduate']}

请用专业、鼓励的语气，给出具体可操作的建议。`;

    this.setSystemPrompt(systemPrompt);

    let gradeSummary = '';
    if (grades.length > 0) {
      let totalCredits = 0;
      let totalScore = 0;
      for (let i = 0; i < grades.length; i++) {
        totalCredits += grades[i].credit;
        totalScore += grades[i].score;
      }
      const avgScore = totalScore / grades.length;
      gradeSummary = `平均分: ${avgScore.toFixed(1)}, 总学分: ${totalCredits}`;

      const semesterMap = new Map<string, GradeData[]>();
      for (let i = 0; i < grades.length; i++) {
        const sem = grades[i].semester || '未知学期';
        if (!semesterMap.has(sem)) {
          semesterMap.set(sem, []);
        }
        semesterMap.get(sem)?.push(grades[i]);
      }
    }

    let courseSummary = '';
    if (courses.length > 0) {
      const courseNames: string[] = [];
      for (let i = 0; i < courses.length && i < 10; i++) {
        courseNames.push(courses[i].courseName);
      }
      courseSummary = courseNames.join('、');
    }

    let studentStatus = '';
    if (enrollmentYear && currentSemesterInfo) {
      const studentInfo = SemesterUtils.getStudentCurrentSemester(enrollmentYear);
      if (studentInfo) {
        const gradeName = SemesterUtils.getGradeName(studentInfo.semesterNumber);
        studentStatus = `入学年份: ${enrollmentYear}级
当前学期: ${currentSemesterInfo.displayName}
所在年级: ${gradeName}`;
      }
    }

    const directionLabels: Record<string, string> = {
      'postgraduate': '考研',
      'civil_service': '考公',
      'employment': '就业',
      'abroad': '出国留学'
    };
    const directionLabel = directionLabels[direction] || '综合';

    const prompt = `请为我制定【${directionLabel}】方向的学习规划：

【学生信息】
专业: ${major || '未知'}
${studentStatus}
${gradeSummary ? `成绩概况: ${gradeSummary}` : ''}

【主要课程】
${courseSummary || '暂无课程数据'}

请根据以上信息，提供详细的规划建议，包括：
1. 目标院校/岗位/学校分析
2. 备考/准备时间规划
3. 推荐复习资料和教材
4. 常见面试题目类型
5. 学习资源推荐`;

    return this.chat(prompt);
  }

  async generateQuiz(
    grades: GradeData[],
    courses: TrainingCourseData[],
    major?: string,
    direction: string = 'postgraduate'
  ): Promise<AIResponse> {
    const systemPrompt = `你是一位专业的题库生成专家，擅长根据专业课程生成真实、准确的测试题目。

【重要要求】
1. 题目必须真实、准确、有教育意义
2. 题目应该基于${major || '本专业'}专业核心课程
3. 难度适中，适合大学生水平
4. 每道题必须有明确的正确答案和解析

【输出格式 - 必须严格遵守】
只输出JSON数组，不要有任何其他文字、说明或markdown标记。
输出格式示例：
[{"question":"题目1","options":["选项A","选项B","选项C","选项D"],"correctAnswer":0,"explanation":"解析"},{"question":"题目2","options":["选项A","选项B","选项C","选项D"],"correctAnswer":1,"explanation":"解析"}]

注意：
- correctAnswer是正确答案在options数组中的索引（0表示第一个选项）
- 必须输出完整的JSON数组
- 不要使用markdown代码块
- 不要添加任何说明文字`;

    this.setSystemPrompt(systemPrompt);

    const courseList: string[] = [];
    for (let i = 0; i < courses.length && i < 5; i++) {
      courseList.push(courses[i].courseName);
    }

    const directionContexts: Record<string, string> = {
      'postgraduate': '研究生入学考试',
      'civil_service': '公务员考试行测/申论',
      'employment': '企业面试专业知识',
      'abroad': '留学申请相关'
    };
    const directionContext = directionContexts[direction] || '综合测试';

    const prompt = `请为${major || '本专业'}专业生成5道${directionContext}相关的选择题。

参考课程: ${courseList.join('、') || '专业核心课程'}

要求：
1. 题目应涵盖专业核心知识点
2. 每题4个选项，只有1个正确答案
3. 题目难度适中
4. 必须提供答案解析

现在直接输出JSON数组：`;

    return this.chat(prompt);
  }
}

export interface TrainingCourseData {
  courseName: string;
  courseCode: string;
  credit: number;
  totalHours: number;
  courseType: string;
  courseAttribute: string;
  examType: string;
  semester: string;
}
