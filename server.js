require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('./src/api/student');
const pushService = require('./src/services/pushService');
const notificationMonitor = require('./src/services/notificationMonitor');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

const userPushTokens = new Map();
const userSessions = new Map();

app.get('/', (req, res) => {
    res.json({ message: '教务系统代理服务已启动', status: 'running', mode: 'proxy-only' });
});

app.get('/api/version', (_req, res) => {
    let pkgVersion = 'unknown';
    try {
        const pkg = require('./package.json');
        pkgVersion = pkg && pkg.version ? pkg.version : 'unknown';
    } catch (_e) { }

    res.json({
        name: 'jw-backend',
        version: pkgVersion,
        buildTime: new Date().toISOString(),
        mode: 'proxy-only',
        features: {
            timetableFollowRedirects: true,
            noDatabase: true
        }
    });
});

app.post('/api/sync', async (req, res) => {
    const { username, password, semester } = req.body;

    if (!username || !password) {
        return res.status(400).json({ success: false, message: '请提供学号和密码' });
    }

    try {
        console.log(`正在获取学生数据: ${username}${semester ? ` (学期: ${semester})` : ''}`);
        const loginResult = await login(username, password);

        if (!loginResult.success) {
            return res.status(401).json({ success: false, message: loginResult.message });
        }

        const cookies = loginResult.cookies;

        const info = await getStudentInfo(cookies);
        if (!info) {
            return res.status(500).json({ success: false, message: '获取学生信息失败' });
        }

        let currentSemester = semester;
        const timetable = await getTimetable(cookies, semester);
        if (timetable && timetable.length > 0 && timetable[0] && timetable[0].semester) {
            currentSemester = timetable[0].semester;
            console.log(`从课表中获取到当前学期: ${currentSemester}`);
        }

        const [grades, exams, plans, progress] = await Promise.all([
            getGrades(cookies).catch(e => {
                console.error('获取成绩失败:', e);
                return null;
            }),
            getExamSchedule(cookies, currentSemester).catch(e => {
                console.error('获取考试安排失败:', e);
                return null;
            }),
            getSemesterPlan(cookies).catch(e => {
                console.error('获取培养计划失败:', e);
                return null;
            }),
            getStudyProgress(cookies).catch(e => {
                console.error('获取学分进度失败:', e);
                return null;
            })
        ]);

        const timetableCount = timetable ? timetable.length : 0;
        let gradesCount = 0;
        if (grades) {
            for (const sem in grades) {
                if (grades[sem] && Array.isArray(grades[sem])) {
                    gradesCount += grades[sem].length;
                }
            }
        }

        console.log(`数据获取完成: ${username}`);
        console.log(`数据统计: 课表=${timetableCount}, 成绩=${gradesCount}, 考试=${exams ? exams.length : 0}, 培养计划=${plans ? Object.keys(plans).length : 0}学期, 学分进度=${progress ? progress.length : 0}`);

        userSessions.set(username, {
            cookies: cookies,
            lastSync: Date.now()
        });

        return res.json({
            success: true,
            message: '数据获取成功',
            data: {
                info,
                courses: timetable || [],
                grades: grades || {},
                exams: exams || [],
                plans: plans || {},
                progress: progress || [],
                semester: currentSemester,
                lastUpdated: Date.now()
            }
        });
    } catch (error) {
        console.error('Sync error:', error);
        const errorMessage = error && error.message ? error.message : String(error);
        res.status(500).json({
            success: false,
            message: '服务器内部错误: ' + errorMessage
        });
    }
});

app.get('/api/semester/latest', async (_req, res) => {
    res.json({
        success: true,
        data: {
            semester: new Date().getFullYear() + '-' + (new Date().getMonth() >= 8 ? '1' : '2')
        }
    });
});

app.post('/api/push/register', async (req, res) => {
    const { studentId, pushToken } = req.body;

    if (!studentId || !pushToken) {
        return res.status(400).json({ success: false, message: '请提供学号和推送Token' });
    }

    userPushTokens.set(studentId, {
        token: pushToken,
        registeredAt: Date.now()
    });

    const session = userSessions.get(studentId);
    if (session) {
        notificationMonitor.registerUser(studentId, session.cookies, pushToken);
    }

    console.log(`Push token registered for student: ${studentId}`);
    res.json({ success: true, message: '推送Token注册成功' });
});

app.post('/api/push/unregister', async (req, res) => {
    const { studentId } = req.body;

    if (!studentId) {
        return res.status(400).json({ success: false, message: '请提供学号' });
    }

    userPushTokens.delete(studentId);
    notificationMonitor.unregisterUser(studentId);
    console.log(`Push token unregistered for student: ${studentId}`);
    res.json({ success: true, message: '推送Token注销成功' });
});

app.post('/api/push/test', async (req, res) => {
    const { studentId, type, title, content } = req.body;

    if (!studentId) {
        return res.status(400).json({ success: false, message: '请提供学号' });
    }

    const tokenInfo = userPushTokens.get(studentId);
    if (!tokenInfo) {
        return res.status(404).json({ success: false, message: '未找到该用户的推送Token' });
    }

    const result = await pushService.sendPushNotification(
        tokenInfo.token,
        title || '测试通知',
        content || '这是一条测试消息',
        type || 'course_change'
    );

    res.json(result);
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT} (proxy-only mode, no database)`);
    console.log(`Push notifications: ${process.env.HUAWEI_CLIENT_ID ? 'enabled' : 'disabled (no credentials)'}`);
    
    if (process.env.HUAWEI_CLIENT_ID) {
        notificationMonitor.startMonitoring();
        console.log('Notification monitoring service started');
    }
});
