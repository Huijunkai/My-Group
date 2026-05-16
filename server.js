require('dotenv').config({ path: __dirname + '/.env' });
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const fs = require('fs');
const { isMockMode, getModeInfo } = require('./src/mode');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('./src/api/student');
const { getAnnouncements, getAnnouncementDetail } = require('./src/api/announcement');
const { getCampuses, getBuildings, queryEmptyRooms, queryRoomSchedule } = require('./src/api/emptyroom');
const { scanWaterQrcode, initWaterDevice, parseScanUrl, bindWaterAccount, getWaterBalance } = require('./src/api/water');
const { getElectricity, saveElectricityReminderSettings, getElectricityReminderSettings } = require('./src/api/electricity');
const xyyxt = require('./src/xyyxt');
const pushService = require('./src/services/pushService');
const notificationMonitor = require('./src/services/notificationMonitor');
const electricityMonitor = require('./src/services/electricityMonitor');
const { initDatabase, checkHealth } = require('./src/db');
const { syncStudent, syncCourses, syncGrades, syncExams, syncPlans, syncProgress } = require('./src/db/sync');
const { getEncryptionKeyBase64, getIvBase64 } = require('./src/utils/encryption');
const { UserPushToken, Grade, Exam, Course } = require('./src/db/models');
const { generateToken, authenticate, optionalAuth } = require('./src/middleware/auth');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
const mockMode = isMockMode();

console.log(`========================================`);
console.log(`教务系统后端服务启动`);
console.log(`运行模式: ${mockMode ? '模拟数据模式 (MOCK)' : '生产环境模式 (PRODUCTION)'}`);
console.log(`环境变量 MOCK_MODE: ${process.env.MOCK_MODE || '未设置'}`);
console.log(`========================================`);

const userPushTokens = new Map();
const userSessions = new Map();

app.get('/', (req, res) => {
    const modeInfo = getModeInfo();
    res.json({ 
        message: `教务系统代理服务已启动（${modeInfo.isMock ? '模拟数据模式' : '生产环境模式'}）`, 
        status: 'running', 
        mode: modeInfo.mode,
        ...modeInfo
    });
});

app.get('/api/version', (_req, res) => {
    let pkgVersion = 'unknown';
    try {
        const pkg = require('./package.json');
        pkgVersion = pkg && pkg.version ? pkg.version : 'unknown';
    } catch (_e) { }

    const modeInfo = getModeInfo();

    res.json({
            name: 'jw-backend',
            version: pkgVersion,
            buildTime: new Date().toISOString(),
            mode: modeInfo.mode,
            isMock: modeInfo.isMock,
            features: {
                timetableFollowRedirects: true,
                noDatabase: false,
                encryption: true,
                mockData: modeInfo.isMock
            },
            config: {
                MOCK_MODE: process.env.MOCK_MODE,
                NODE_ENV: process.env.NODE_ENV
            }
    });
});

app.get('/api/mode/info', (req, res) => {
    const modeInfo = getModeInfo();
    
    if (modeInfo.isMock) {
        res.json({
            success: true,
            ...modeInfo,
            message: '当前使用模拟数据模式',
            availableTestAccounts: [
                { studentId: '202101001', password: '123456', name: '张三' },
                { studentId: '202101002', password: '123456', name: '李四' },
                { studentId: '202102001', password: '123456', name: '王五' },
                { studentId: '202103001', password: '123456', name: '赵六' },
                { studentId: '202201001', password: '123456', name: '钱七' }
            ],
            endpoints: {
                auth: '/api/sync (POST)',
                xyyxtAuth: '/api/xyyxt/login (POST)',
                studentInfo: '登录后自动返回',
                dormitory: '/api/xyyxt/buildings, /api/xyyxt/rooms'
            }
        });
    } else {
        res.json({
            success: true,
            ...modeInfo,
            message: '当前使用生产环境模式（真实数据）'
        });
    }
});

app.get('/api/encryption/key', (_req, res) => {
    res.json({
        success: true,
        data: {
            key: getEncryptionKeyBase64(),
            iv: getIvBase64(),
            algorithm: 'AES-256-CBC'
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

        try {
            const dbOk = await checkHealth();
            if (!dbOk) {
                console.warn(`数据库不可用，跳过数据同步: ${username}`);
            } else {
                console.log(`开始同步数据到数据库: ${username}`);
                
                const realtimePush = require('./src/services/realtimePush');
                
                const gradeResults = await syncGrades(username, grades || {});
                if (gradeResults && gradeResults.length > 0) {
                    console.log(`检测到 ${gradeResults.length} 条新成绩，触发实时推送`);
                    for (const result of gradeResults) {
                        if (result.success) {
                            await realtimePush.notifyNewGradeRealtime(username, result.grade);
                        }
                    }
                }
                
                const examResults = await syncExams(username, exams || []);
                if (examResults && examResults.length > 0) {
                    console.log(`检测到 ${examResults.length} 条新考试，触发实时推送`);
                    for (const result of examResults) {
                        if (result.success) {
                            await realtimePush.notifyNewExamRealtime(username, result.exam);
                        }
                    }
                }
                
                await Promise.all([
                    syncStudent(username, info),
                    syncCourses(username, timetable || []),
                    syncPlans(username, plans || {}),
                    syncProgress(username, progress || [])
                ]);
                
                console.log(`数据同步完成: ${username}`);
            }
        } catch (syncError) {
            console.error('数据同步失败:', syncError.message);
        }

        userSessions.set(username, {
            cookies: cookies,
            lastSync: Date.now()
        });

        const jwtToken = generateToken({
            username: username,
            studentId: info.studentId || username
        });

        console.log(`[JWT] 为用户 ${username} 生成新令牌`);

        const gradesArray = grades ? Object.keys(grades).map(semester => ({
            semester,
            grades: grades[semester] || []
        })) : [];

        const plansArray = plans ? Object.keys(plans).map(semester => ({
            semester,
            plans: plans[semester] || []
        })) : [];

        return res.json({
            success: true,
            message: '数据获取成功',
            token: jwtToken,
            data: {
                info,
                courses: timetable || [],
                grades: gradesArray,
                exams: exams || [],
                plans: plansArray,
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

app.post('/api/auth/debug-token', async (req, res) => {
    const { username, studentId } = req.body;

    if (!username && !studentId) {
        return res.status(400).json({
            success: false,
            message: '请提供 username 或 studentId',
            usage: 'POST /api/auth/debug-token { "username": "23490329" }'
        });
    }

    try {
        const token = generateToken({
            username: username || studentId,
            studentId: studentId || username
        });

        console.log(`[DEBUG] 生成测试 Token for user: ${username || studentId}`);

        res.json({
            success: true,
            message: '调试 Token 已生成（仅用于测试，有效期7天）',
            token: token,
            userInfo: {
                username: username || studentId,
                studentId: studentId || username
            },
            expiresIn: '7d',
            curlExample: `curl -X POST http://124.70.92.199:3000/api/push/test \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${token}" \\
  -d '{"studentId": "${studentId || username}", "type": "test", "title": "测试通知", "content": "这是一条测试消息"}'`
        });
    } catch (error) {
        console.error('生成调试 Token 失败:', error);
        res.status(500).json({
            success: false,
            message: '生成 Token 失败: ' + error.message
        });
    }
});

app.post('/api/push/register', authenticate, async (req, res) => {
    const { studentId, pushToken, deviceInfo } = req.body;

    if (!studentId || !pushToken) {
        return res.status(400).json({ success: false, message: '请提供学号和推送Token' });
    }

    if (!pushService.validateToken(pushToken)) {
        console.warn(`[注册拦截] 无效Token被拒绝 学号=${studentId} Token长度=${pushToken.length} 值=${pushToken.substring(0, 20)}`);
        return res.status(400).json({ 
            success: false, 
            message: '推送Token格式无效（不是华为Push Kit真实Token），请确认手机端已正确获取Token',
            tokenLength: pushToken.length,
            hint: '真实华为Push Token通常以AAA开头，当前Token疑似测试/模拟数据'
        });
    }

    try {
        const existingToken = await UserPushToken.findOne({ where: { studentId } });
        
        if (existingToken) {
            existingToken.pushToken = pushToken;
            existingToken.deviceInfo = deviceInfo || existingToken.deviceInfo;
            existingToken.isActive = true;
            existingToken.lastActiveAt = new Date();
            await existingToken.save();
            console.log(`Push token updated for student: ${studentId}`);
        } else {
            await UserPushToken.create({
                studentId,
                pushToken,
                deviceInfo: deviceInfo || 'unknown',
                isActive: true,
                createdAt: new Date(),
                lastActiveAt: new Date()
            });
            console.log(`Push token created for student: ${studentId}`);
        }

        userPushTokens.set(studentId, {
            token: pushToken,
            registeredAt: Date.now()
        });

        const session = userSessions.get(studentId);
        if (session) {
            notificationMonitor.registerUser(studentId, session.cookies, pushToken);
        }

        res.json({ success: true, message: '推送Token注册成功' });
    } catch (error) {
        console.error('Failed to register push token:', error);
        res.status(500).json({ success: false, message: '注册失败: ' + error.message });
    }
});

app.post('/api/push/unregister', authenticate, async (req, res) => {
    const { studentId } = req.body;

    if (!studentId) {
        return res.status(400).json({ success: false, message: '请提供学号' });
    }

    try {
        const userToken = await UserPushToken.findOne({ where: { studentId } });
        if (userToken) {
            userToken.isActive = false;
            await userToken.save();
        }

        userPushTokens.delete(studentId);
        notificationMonitor.unregisterUser(studentId);
        console.log(`Push token unregistered for student: ${studentId}`);
        res.json({ success: true, message: '推送Token注销成功' });
    } catch (error) {
        console.error('Failed to unregister push token:', error);
        res.status(500).json({ success: false, message: '注销失败: ' + error.message });
    }
});

app.post('/api/push/test', authenticate, async (req, res) => {
    const { studentId, type, title, content } = req.body;

    if (!studentId) {
        return res.status(400).json({ success: false, message: '请提供学号' });
    }

    try {
        const userToken = await UserPushToken.findOne({ where: { studentId, isActive: true } });
        if (!userToken) {
            return res.status(404).json({ success: false, message: '未找到该用户的推送Token' });
        }

        const result = await pushService.sendPushNotification(
            userToken.pushToken,
            title || '测试通知',
            content || '这是一条测试消息',
            type || 'course_change'
        );

        res.json(result);
    } catch (error) {
        console.error('Failed to send test push:', error);
        res.status(500).json({ success: false, message: '发送失败: ' + error.message });
    }
});

app.get('/api/push/diagnostics', async (_req, res) => {
    try {
        const allTokens = await UserPushToken.findAll({
            attributes: ['studentId', 'pushToken', 'isActive', 'lastActiveAt', 'createdAt'],
            order: [['lastActiveAt', 'DESC']]
        });

        const diagnostics = allTokens.map(t => {
            const token = t.pushToken || '';
            const valid = pushService.isValidPushToken(token);
            return {
                studentId: t.studentId,
                tokenLength: token.length,
                tokenPrefix: token.substring(0, 15) + (token.length > 15 ? '...' : ''),
                isValid: valid,
                isActive: t.isActive,
                registeredAt: t.createdAt,
                lastActiveAt: t.lastActiveAt
            };
        });

        res.json({
            success: true,
            totalUsers: diagnostics.length,
            validTokens: diagnostics.filter(d => d.isValid).length,
            invalidTokens: diagnostics.filter(d => !d.isValid).length,
            details: diagnostics
        });
    } catch (error) {
        console.error('Push diagnostics error:', error);
        res.status(500).json({ success: false, message: '诊断失败' });
    }
});

app.post('/api/push/cleanup', async (_req, res) => {
    try {
        const allTokens = await UserPushToken.findAll({
            attributes: ['id', 'studentId', 'pushToken']
        });

        let cleaned = 0;
        const removed = [];

        for (const t of allTokens) {
            if (!pushService.isValidPushToken(t.pushToken)) {
                await t.destroy();
                cleaned++;
                removed.push({
                    studentId: t.studentId,
                    tokenLength: t.pushToken.length,
                    tokenPrefix: t.pushToken.substring(0, 20)
                });
                userPushTokens.delete(t.studentId);
                notificationMonitor.unregisterUser(t.studentId);
            }
        }

        console.log(`[Token清理] 清理完成，删除 ${cleaned} 个无效Token`);
        res.json({
            success: true,
            message: `已清理 ${cleaned} 个无效推送Token`,
            cleanedCount: cleaned,
            removed
        });
    } catch (error) {
        console.error('Push cleanup error:', error);
        res.status(500).json({ success: false, message: '清理失败' });
    }
});

app.get('/api/push/test-huawei', async (_req, res) => {
    try {
        const axios = require('axios');
        const projectId = process.env.HUAWEI_PROJECT_ID;
        const clientId = process.env.HUAWEI_CLIENT_ID;

        let tokenResult = { success: false };
        try {
            const tokenUrl = 'https://oauth-login.cloud.huawei.com/oauth2/v3/token';
            const params = new URLSearchParams();
            params.append('grant_type', 'client_credentials');
            params.append('client_id', clientId);
            params.append('client_secret', process.env.HUAWEI_CLIENT_SECRET);

            const tokenRes = await axios.post(tokenUrl, params, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                timeout: 10000
            });

            if (tokenRes.data.access_token) {
                tokenResult = {
                    success: true,
                    tokenPrefix: tokenRes.data.access_token.substring(0, 20) + '...',
                    expiresIn: tokenRes.data.expires_in,
                    scope: tokenRes.data.scope || '(未返回scope)'
                };

                const testUrl = `https://push-api.cloud.huawei.com/v1/${projectId}/messages:send`;
                const testPayload = {
                    validate_only: true,
                    message: {
                        android: {
                            notification: { title: '诊断测试', body: '这是一条验证消息' },
                            ttl: '60s'
                        },
                        token: ['TEST_DIAGNOSTIC_TOKEN'],
                        data: { diagnostic: true }
                    }
                };

                try {
                    const pushRes = await axios.post(testUrl, testPayload, {
                        headers: {
                            'Authorization': `Bearer ${tokenRes.data.access_token}`,
                            'Content-Type': 'application/json'
                        },
                        timeout: 10000
                    });

                    return res.json({
                        success: true,
                        phase: 'complete',
                        oauth: tokenResult,
                        pushApi: {
                            url: testUrl,
                            projectId: projectId,
                            oauthClientId: clientId.substring(0, 10) + '...',
                            responseCode: pushRes.data.code,
                            responseMsg: pushRes.data.msg,
                            requestId: pushRes.data.request_id || null,
                            isTokenError: String(pushRes.data.code).includes('803'),
                            analysis: {
                                code80000000: pushRes.data.code === '80000000' ? '✅ 推送权限正常，问题在Token' : '',
                                code80300002: pushRes.data.code === '80300002' ? '❌ OAuth客户端无权推送 → 检查OAuth授权列表是否有Push Kit scope' : '',
                                code80100001: pushRes.data.code === '80100001' ? '❌ Token无效' : '',
                                other: !['80000000','80300002','80100001'].includes(pushRes.data.code) ? `⚠️ 未知错误码: ${pushRes.data.code}` : ''
                            }
                        }
                    });
                } catch (pushErr) {
                    return res.json({
                        success: false,
                        phase: 'push_api_call_failed',
                        oauth: tokenResult,
                        pushApi: {
                            url: testUrl,
                            httpStatus: pushErr.response?.status,
                            errorCode: pushErr.response?.data?.code,
                            errorMsg: pushErr.response?.data?.msg,
                            errorDetail: JSON.stringify(pushErr.response?.data).substring(0, 500)
                        },
                        suggestion: pushErr.response?.data?.code === '80300002'
                            ? '❌ 确认: OAuth客户端缺少Push Kit权限。去 用户与访问→OAuth客户端→青序→授权 列表确认'
                            : '查看上方错误详情'
                    });
                }
            } else {
                tokenResult = { success: false, rawResponse: JSON.stringify(tokenRes.data).substring(0, 300) };
            }
        } catch (oauthErr) {
            tokenResult = {
                success: false,
                error: oauthErr.message,
                httpStatus: oauthErr.response?.status,
                errorBody: JSON.stringify(oauthErr.response?.data).substring(0, 300)
            };
        }

        return res.json({
            success: false,
            phase: 'oauth_failed',
            oauth: tokenResult,
            config: {
                projectId: projectId || '未设置',
                clientId: clientId ? clientId.substring(0, 10) + '...' : '未设置',
                hasKeyFile: !!process.env.HUAWEI_KEY_FILE
            },
            suggestion: !clientId ? '❌ .env 中 HUAWEI_CLIENT_ID 未设置'
                : !projectId ? '❌ .env 中 HUAWEI_PROJECT_ID 未设置'
                : '❌ OAuth认证失败，检查 client_id 和 client_secret 是否正确'
        });
    } catch (error) {
        res.status(500).json({ success: false, message: '诊断失败', error: error.message });
    }
});

app.get('/api/announcements', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 5;
        const offset = parseInt(req.query.offset) || 0;
        const result = await getAnnouncements(limit, offset);
        res.json({
            success: true,
            data: result.announcements,
            total: result.total
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

app.post('/api/water/scan', async (req, res) => {
    const { scanUrl } = req.body;

    if (!scanUrl) {
        return res.status(400).json({ success: false, message: '请提供扫描链接' });
    }

    try {
        console.log(`打水系统: 扫码请求 - ${scanUrl.substring(0, 80)}...`);
        const result = await bindWaterAccount(scanUrl);
        
        if (result.success) {
            return res.json({
                success: true,
                message: '绑定成功',
                data: result.data
            });
        } else {
            return res.status(400).json({
                success: false,
                message: result.message
            });
        }
    } catch (error) {
        console.error('打水系统扫码失败:', error);
        res.status(500).json({
            success: false,
            message: '扫码失败: ' + error.message
        });
    }
});

app.post('/api/water/init', async (req, res) => {
    const { openid, deviceid, app } = req.body;

    if (!openid || !deviceid) {
        return res.status(400).json({ success: false, message: '请提供 openid 和 deviceid' });
    }

    try {
        console.log(`打水系统: 初始化设备 ${deviceid}`);
        const result = await initWaterDevice(openid, deviceid, app || 'WECHAT');
        
        if (result.success) {
            return res.json({
                success: true,
                message: '设备初始化成功',
                data: result.data
            });
        } else {
            return res.status(400).json({
                success: false,
                message: result.message
            });
        }
    } catch (error) {
        console.error('打水系统初始化失败:', error);
        res.status(500).json({
            success: false,
            message: '初始化失败: ' + error.message
        });
    }
});

app.post('/api/water/parse', async (req, res) => {
    const { scanUrl } = req.body;

    if (!scanUrl) {
        return res.status(400).json({ success: false, message: '请提供扫描链接' });
    }

    try {
        const result = parseScanUrl(scanUrl);
        
        if (result.success) {
            return res.json({
                success: true,
                data: result.data
            });
        } else {
            return res.status(400).json({
                success: false,
                message: result.message
            });
        }
    } catch (error) {
        res.status(500).json({
            success: false,
            message: '解析失败: ' + error.message
        });
    }
});

app.post('/api/water/balance', async (req, res) => {
    const { openid, saler, app } = req.body;

    if (!openid) {
        return res.status(400).json({ success: false, message: '请提供 openid' });
    }

    try {
        console.log(`打水系统: 获取余额 - openid: ${openid}`);
        const result = await getWaterBalance(openid, saler || '', app || 'WECHAT');
        
        if (result.success) {
            return res.json({
                success: true,
                message: '获取余额成功',
                data: result.data
            });
        } else {
            return res.status(400).json({
                success: false,
                message: result.message
            });
        }
    } catch (error) {
        console.error('打水系统获取余额失败:', error);
        res.status(500).json({
            success: false,
            message: '获取余额失败: ' + error.message
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

app.post('/api/electricity/reminder/settings', async (req, res) => {
    const { studentId, settings } = req.body;

    if (!studentId || !settings) {
        return res.status(400).json({ success: false, message: '请提供学号和设置信息' });
    }

    try {
        const result = await saveElectricityReminderSettings(studentId, settings);
        res.json(result);
    } catch (error) {
        console.error('保存电费提醒设置失败:', error);
        res.status(500).json({
            success: false,
            message: '保存失败: ' + error.message
        });
    }
});

app.get('/api/electricity/reminder/settings', async (req, res) => {
    const { studentId } = req.query;

    if (!studentId) {
        return res.status(400).json({ success: false, message: '请提供学号' });
    }

    try {
        const result = await getElectricityReminderSettings(studentId);
        res.json(result);
    } catch (error) {
        console.error('获取电费提醒设置失败:', error);
        res.status(500).json({
            success: false,
            message: '获取失败: ' + error.message
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

async function startServer() {
    try {
        // 尝试初始化数据库
        try {
            const dbInitialized = await initDatabase();
            console.log('数据库初始化完成，状态:', dbInitialized ? '成功' : '失败（无数据库模式）');
        } catch (dbError) {
            console.warn('数据库初始化失败:', dbError.message);
            console.warn('将以无数据库模式启动，部分功能可能不可用');
        }
        
        // 启动服务器
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`Server is running on port ${PORT}`);
            
            const hasJwtConfig = process.env.HUAWEI_PRIVATE_KEY_FILE && fs.existsSync(process.env.HUAWEI_PRIVATE_KEY_FILE);
            console.log(`Push notifications: ${hasJwtConfig ? 'enabled (JWT)' : 'disabled (no JWT config)'}`);
            
            if (hasJwtConfig) {
                notificationMonitor.startMonitoring();
                console.log('Notification monitoring service started');
                
                try {
                    electricityMonitor.start();
                    console.log('Electricity monitoring service started');
                } catch (e) {
                    console.warn('启动电费监控服务失败:', e.message);
                }
            }
        });
    } catch (error) {
        console.error('启动服务器失败:', error.message);
        process.exit(1);
    }
}

startServer();
