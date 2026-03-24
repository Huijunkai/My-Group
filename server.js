require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('./src/api/student');
const { getAnnouncements, getAnnouncementDetail } = require('./src/api/announcement');
const { getCampuses, getBuildings, queryEmptyRooms, queryRoomSchedule } = require('./src/api/emptyroom');
const xyyxt = require('./src/xyyxt');
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

app.get('/api/announcements', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 5;
        const announcements = await getAnnouncements(limit);
        res.json({
            success: true,
            data: announcements
        });
    } catch (error) {
        console.error('获取公告失败:', error);
        res.status(500).json({
            success: false,
            message: '获取公告失败: ' + error.message
        });
    }
});

app.get('/api/emptyroom/campuses', async (_req, res) => {
    try {
        const campuses = await getCampuses();
        res.json({
            success: true,
            data: campuses
        });
    } catch (error) {
        console.error('获取校区失败:', error);
        res.status(500).json({
            success: false,
            message: '获取校区失败: ' + error.message
        });
    }
});

app.get('/api/emptyroom/buildings', async (req, res) => {
    try {
        const { campus } = req.query;
        
        let cookies = null;
        for (const [sid, sess] of userSessions) {
            if (sess.cookies) {
                cookies = sess.cookies;
                break;
            }
        }
        
        const buildings = await getBuildings(cookies, campus);
        res.json({
            success: true,
            data: buildings
        });
    } catch (error) {
        console.error('获取教学楼失败:', error);
        res.status(500).json({
            success: false,
            message: '获取教学楼失败: ' + error.message
        });
    }
});

app.post('/api/emptyroom/query', async (req, res) => {
    try {
        const { semester, campus, building, weekStart, weekEnd, periodStart, periodEnd } = req.body;
        
        if (!semester) {
            return res.status(400).json({
                success: false,
                message: '请提供学期参数'
            });
        }

        const studentId = req.body.studentId || 'guest';
        let cookies = null;
        
        const session = userSessions.get(studentId);
        if (session && session.cookies) {
            cookies = session.cookies;
        } else {
            for (const [sid, sess] of userSessions) {
                if (sess.cookies) {
                    cookies = sess.cookies;
                    break;
                }
            }
        }
        
        if (!cookies) {
            return res.status(401).json({
                success: false,
                message: '请先登录'
            });
        }

        const rooms = await queryEmptyRooms(cookies, {
            semester,
            campus,
            building,
            weekStart,
            weekEnd,
            periodStart,
            periodEnd
        });
        
        res.json({
            success: true,
            data: rooms
        });
    } catch (error) {
        console.error('查询空教室失败:', error);
        res.status(500).json({
            success: false,
            message: '查询空教室失败: ' + error.message
        });
    }
});

app.post('/api/emptyroom/schedule', async (req, res) => {
    try {
        const { roomName, semester, campus, building, weekStart, weekEnd, periodStart, periodEnd } = req.body;
        
        if (!roomName) {
            return res.status(400).json({
                success: false,
                message: '请提供教室名称'
            });
        }

        const studentId = req.body.studentId || 'guest';
        let cookies = null;
        
        const session = userSessions.get(studentId);
        if (session && session.cookies) {
            cookies = session.cookies;
        } else {
            for (const [sid, sess] of userSessions) {
                if (sess.cookies) {
                    cookies = sess.cookies;
                    break;
                }
            }
        }
        
        if (!cookies) {
            return res.status(401).json({
                success: false,
                message: '请先登录'
            });
        }

        const schedule = await queryRoomSchedule(cookies, {
            roomName,
            semester,
            campus,
            building,
            weekStart,
            weekEnd,
            periodStart,
            periodEnd
        });
        
        if (schedule) {
            res.json({
                success: true,
                data: schedule
            });
        } else {
            res.json({
                success: false,
                message: '未找到该教室'
            });
        }
    } catch (error) {
        console.error('查询教室课表失败:', error);
        res.status(500).json({
            success: false,
            message: '查询教室课表失败: ' + error.message
        });
    }
});

app.get('/api/announcements/detail', async (req, res) => {
    try {
        const { url } = req.query;
        if (!url) {
            return res.status(400).json({
                success: false,
                message: '请提供公告URL'
            });
        }
        const detail = await getAnnouncementDetail(url);
        if (detail) {
            res.json({
                success: true,
                data: detail
            });
        } else {
            res.status(404).json({
                success: false,
                message: '未找到公告详情'
            });
        }
    } catch (error) {
        console.error('获取公告详情失败:', error);
        res.status(500).json({
            success: false,
            message: '获取公告详情失败: ' + error.message
        });
    }
});

app.post('/api/xyyxt/login', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ success: false, message: '请提供账号和密码' });
    }

    try {
        console.log(`校园一信通登录: ${username}`);
        const result = await xyyxt.login(username, password);

        if (result.success) {
            userSessions.set(`xyyxt_${username}`, {
                accessToken: result.data.access_token,
                refreshToken: result.data.refresh_token,
                schoolId: result.data.schoolId,
                userId: result.data.userId,
                lastSync: Date.now()
            });

            return res.json({
                success: true,
                message: '登录成功',
                data: result.data
            });
        } else {
            return res.status(401).json({
                success: false,
                message: result.message || '登录失败'
            });
        }
    } catch (error) {
        console.error('校园一信通登录错误:', error);
        res.status(500).json({
            success: false,
            message: '登录失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/userinfo', async (req, res) => {
    const { username } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const userInfo = await xyyxt.getUserInfo(session.accessToken);
        res.json({
            success: true,
            data: userInfo
        });
    } catch (error) {
        console.error('获取用户信息失败:', error);
        res.status(500).json({
            success: false,
            message: '获取用户信息失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/balance', async (req, res) => {
    const { username } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const balance = await xyyxt.getBalance(session.accessToken);
        res.json({
            success: true,
            data: balance
        });
    } catch (error) {
        console.error('获取余额失败:', error);
        res.status(500).json({
            success: false,
            message: '获取余额失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/transactions', async (req, res) => {
    const { username, page, size } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const transactions = await xyyxt.getTransactions(
            session.accessToken,
            parseInt(page) || 1,
            parseInt(size) || 20
        );
        res.json({
            success: true,
            data: transactions
        });
    } catch (error) {
        console.error('获取交易记录失败:', error);
        res.status(500).json({
            success: false,
            message: '获取交易记录失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/consumption', async (req, res) => {
    const { username, page, size } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const consumption = await xyyxt.getConsumptionRecords(
            session.accessToken,
            parseInt(page) || 1,
            parseInt(size) || 20
        );
        res.json({
            success: true,
            data: consumption.data,
            total: consumption.total,
            pages: consumption.pages,
            current: consumption.current
        });
    } catch (error) {
        console.error('获取消费记录失败:', error);
        res.status(500).json({
            success: false,
            message: '获取消费记录失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/recharge', async (req, res) => {
    const { username, page, size } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const recharge = await xyyxt.getRechargeRecords(
            session.accessToken,
            parseInt(page) || 1,
            parseInt(size) || 20
        );
        res.json({
            success: true,
            data: recharge.data,
            total: recharge.total,
            pages: recharge.pages,
            current: recharge.current
        });
    } catch (error) {
        console.error('获取充值记录失败:', error);
        res.status(500).json({
            success: false,
            message: '获取充值记录失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/buildings', async (req, res) => {
    const { username, areaId } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const buildings = await xyyxt.getBuildings(session.accessToken, areaId);
        res.json({
            success: true,
            data: buildings
        });
    } catch (error) {
        console.error('获取宿舍楼失败:', error);
        res.status(500).json({
            success: false,
            message: '获取宿舍楼失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/rooms', async (req, res) => {
    const { username, buildingId, areaId } = req.query;

    if (!username || !buildingId) {
        return res.status(400).json({ success: false, message: '请提供账号和楼栋ID' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const rooms = await xyyxt.getRooms(session.accessToken, buildingId, areaId);
        res.json({
            success: true,
            data: rooms
        });
    } catch (error) {
        console.error('获取宿舍房间失败:', error);
        res.status(500).json({
            success: false,
            message: '获取宿舍房间失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/rooms/all', async (req, res) => {
    const { username, buildingId, areaId } = req.query;

    if (!username) {
        return res.status(400).json({ success: false, message: '请提供账号' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        if (buildingId) {
            const rooms = await xyyxt.getAllRoomsByBuilding(session.accessToken, buildingId, areaId || 'nnxq');
            res.json({
                success: true,
                data: rooms
            });
        } else {
            const allRooms = await xyyxt.getAllBuildingsRooms(session.accessToken, areaId || 'nnxq');
            res.json({
                success: true,
                data: allRooms
            });
        }
    } catch (error) {
        console.error('获取所有房间失败:', error);
        res.status(500).json({
            success: false,
            message: '获取所有房间失败: ' + error.message
        });
    }
});

app.get('/api/xyyxt/electricity', async (req, res) => {
    const { username, roomId, areaId, buildingId } = req.query;

    if (!username || !roomId) {
        return res.status(400).json({ success: false, message: '请提供账号和房间ID' });
    }

    const session = userSessions.get(`xyyxt_${username}`);
    if (!session) {
        return res.status(401).json({ success: false, message: '请先登录' });
    }

    try {
        const electricity = await xyyxt.getElectricity(session.accessToken, roomId, areaId || '', buildingId || '');
        res.json({
            success: true,
            data: electricity
        });
    } catch (error) {
        console.error('获取电费余额失败:', error);
        res.status(500).json({
            success: false,
            message: '获取电费余额失败: ' + error.message
        });
    }
});

app.post('/api/ai/chat', async (req, res) => {
    const { messages, apiKey, model } = req.body;

    if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ success: false, message: '请提供消息列表' });
    }

    if (!apiKey) {
        return res.status(400).json({ success: false, message: '请提供 API Key' });
    }

    const selectedModel = model || 'MiniMax-M2.5';
    const SCNET_URL = 'https://api.scnet.cn/api/llm/v1/chat/completions';

    try {
        const requestBody = {
            model: selectedModel,
            messages: messages,
            temperature: 0.7
        };

        console.log(`AI Proxy: Requesting scnet (${selectedModel})...`);
        console.log(`AI Proxy: API Key (first 10 chars): ${apiKey ? apiKey.substring(0, 10) + '...' : 'EMPTY'}`);
        
        const response = await fetch(SCNET_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`AI Proxy: API error ${response.status}: ${errorText}`);
            
            let errorMessage = `AI API 错误: ${response.status}`;
            try {
                const errorJson = JSON.parse(errorText);
                if (errorJson.error?.message) {
                    errorMessage = errorJson.error.message;
                } else if (errorJson.message) {
                    errorMessage = errorJson.message;
                }
            } catch (e) {
                // 无法解析为 JSON，使用默认消息
            }
            
            return res.json({ 
                success: false, 
                message: errorMessage,
                details: errorText,
                statusCode: response.status
            });
        }

        const result = await response.json();
        console.log(`AI Proxy: Success from scnet (${selectedModel})`);

        const content = result.choices?.[0]?.message?.content || '';

        res.json({ success: true, content: content });
    } catch (error) {
        console.error('AI Proxy error:', error);
        res.status(500).json({ 
            success: false, 
            message: 'AI 请求失败: ' + (error.message || String(error))
        });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT} (proxy-only mode, no database)`);
    console.log(`Push notifications: ${process.env.HUAWEI_CLIENT_ID ? 'enabled' : 'disabled (no credentials)'}`);
    
    if (process.env.HUAWEI_CLIENT_ID) {
        notificationMonitor.startMonitoring();
        console.log('Notification monitoring service started');
    }
});
