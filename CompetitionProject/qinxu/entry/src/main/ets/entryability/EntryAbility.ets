import { AbilityConstant, ConfigurationConstant, UIAbility, Want, Configuration } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { DataStore } from '../utils/DataStore';
import { preferences } from '@kit.ArkData';
import { ThemeColorManager } from '../utils/ThemeColorManager';
import { LocalDataManager } from '../utils/LocalDataManager';
import { NotificationManager } from '../utils/NotificationManager';
import { CourseReminderService } from '../services/CourseReminderService';
import { PushNotificationService } from '../services/PushNotificationService';
import router from '@ohos.router';

const DOMAIN = 0x0000;
const PREFERENCES_NAME = 'theme_preferences';
const KEY_THEME_MODE = 'themeMode';
const KEY_AI_API_KEY = 'ai_api_key';
const KEY_AI_MODEL = 'ai_model';

export default class EntryAbility extends UIAbility {
  private storedLaunchWant: Want | null = null;

  async onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): Promise<void> {
    this.storedLaunchWant = want;
    try {
      const dataStore = DataStore.getInstance();
      await dataStore.initialize(this.context);

      const themeColorManager = ThemeColorManager.getInstance();
      const localDataManager = LocalDataManager.getInstance();
      const notificationManager = NotificationManager.getInstance();
      const courseReminderService = CourseReminderService.getInstance();
      const pushNotificationService = PushNotificationService.getInstance();

      await Promise.all([
        this.loadAIConfig(),
        themeColorManager.initialize(this.context),
        localDataManager.initialize(this.context),
        notificationManager.initialize(this.context),
        courseReminderService.initialize(this.context),
        pushNotificationService.initialize(this.context).then(() => {
          pushNotificationService.registerMessageReceiver(this);
        })
      ]);

      await this.applyThemeMode();

    } catch (err) {
      hilog.error(DOMAIN, 'testTag', 'Failed to initialize. Cause: %{public}s', JSON.stringify(err));
    }
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onCreate');
  }

  private async applyThemeMode(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      const savedTheme = await dataPreferences.get(KEY_THEME_MODE, 'system') as string;
      
      AppStorage.SetOrCreate('themeMode', savedTheme);
      
      let colorMode: ConfigurationConstant.ColorMode;
      switch (savedTheme) {
        case 'light':
          colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_LIGHT;
          break;
        case 'dark':
          colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
          break;
        default:
          colorMode = ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET;
          break;
      }
      
      this.context.getApplicationContext().setColorMode(colorMode);
      hilog.info(DOMAIN, 'testTag', 'Applied theme mode: %{public}s', savedTheme);
      
    } catch (err) {
      hilog.error(DOMAIN, 'testTag', 'Failed to apply theme mode. Cause: %{public}s', JSON.stringify(err));
      this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    }
  }

  /**
   * Ability销毁时回调
   */
  onDestroy(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onDestroy');
    try {
      const courseReminderService = CourseReminderService.getInstance();
      courseReminderService.removeNotificationCancelListener();
    } catch (err) {
      hilog.error(DOMAIN, 'testTag', 'Failed to remove notification listener: %{public}s', JSON.stringify(err));
    }
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onWindowStageCreate');

    // 检查是否有从提醒通知跳转过来的意图
    const want: Want | null = this.storedLaunchWant;
    if (want && want.parameters) {
      const reminderSource = want.parameters['reminderSource'] as string;
      const targetPage = want.parameters['targetPage'] as string;
      
      if (reminderSource === 'course_reminder') {
        hilog.info(DOMAIN, 'testTag', 'Launched from course reminder, target page: %{public}s', targetPage);
        
        let targetContent = 'pages/SchedulePage';
        if (targetPage === 'HomePage') {
          targetContent = 'pages/SchedulePage';
        }
        
        windowStage.loadContent(targetContent, (err) => {
          if (err.code) {
            hilog.error(DOMAIN, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
            return;
          }
          hilog.info(DOMAIN, 'testTag', 'Succeeded in loading reminder target content: %{public}s', targetContent);
        });
      } else {
        // 正常启动，加载首页
        windowStage.loadContent('pages/Index', (err) => {
          if (err.code) {
            hilog.error(DOMAIN, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
            return;
          }
          hilog.info(DOMAIN, 'testTag', 'Succeeded in loading the content.');
        });
      }
    } else {
      // 正常启动，加载首页
      windowStage.loadContent('pages/Index', (err) => {
        if (err.code) {
          hilog.error(DOMAIN, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
          return;
        }
        hilog.info(DOMAIN, 'testTag', 'Succeeded in loading the content.');
      });
    }

    windowStage.getMainWindow((err, mainWindow) => {
      if (err.code) {
        hilog.error(DOMAIN, 'testTag', 'Failed to get main window. Cause: %{public}s', JSON.stringify(err));
        return;
      }

      mainWindow.setWindowLayoutFullScreen(true).then(() => {
        hilog.info(DOMAIN, 'testTag', 'Succeeded in setting window layout full screen.');
      }).catch((err: Error) => {
        hilog.error(DOMAIN, 'testTag', 'Failed to set window layout full screen. Cause: %{public}s', JSON.stringify(err));
      });
    });
  }

  /**
   * 窗口阶段销毁时回调
   */
  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onForeground');
    this.refreshRemindersIfNeeded();
  }

  onBackground(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onBackground');
  }

  onConfigurationUpdate(config: Configuration): void {
    hilog.info(DOMAIN, 'testTag', 'Configuration updated: %{public}s', JSON.stringify(config));
    
    const isDarkMode = config.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
    AppStorage.SetOrCreate('isDarkMode', isDarkMode);
    hilog.info(DOMAIN, 'testTag', 'Dark mode updated: %{public}s', String(isDarkMode));
  }

  /**
   * 刷新提醒服务（如需要）
   * 在应用进入前台时刷新课程提醒
   */
  private async refreshRemindersIfNeeded(): Promise<void> {
    try {
      const courseReminderService = CourseReminderService.getInstance();
      await courseReminderService.refreshRemindersOnForeground();
      hilog.info(DOMAIN, 'testTag', 'Reminders refreshed on foreground');
    } catch (err) {
      hilog.error(DOMAIN, 'testTag', 'Failed to refresh reminders: %{public}s', JSON.stringify(err));
    }
  }

  /**
   * 加载AI配置
   * 从本地存储读取API Key和模型设置
   */
  private async loadAIConfig(): Promise<void> {
    try {
      const dataPreferences = await preferences.getPreferences(this.context, PREFERENCES_NAME);
      const savedApiKey = await dataPreferences.get(KEY_AI_API_KEY, '') as string;
      const savedModel = await dataPreferences.get(KEY_AI_MODEL, 'MiniMax-M2.5') as string;
      
      AppStorage.SetOrCreate('ai_api_key', savedApiKey);
      AppStorage.SetOrCreate('ai_model', savedModel);
      
      hilog.info(DOMAIN, 'testTag', 'AI config loaded');
    } catch (err) {
      hilog.error(DOMAIN, 'testTag', 'Failed to load AI config: %{public}s', JSON.stringify(err));
    }
  }
}