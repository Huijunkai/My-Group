const axios = require('axios');
const fs = require('fs');
const jwt = require('jsonwebtoken');

let cachedToken = null;
let tokenExpireTime = 0;
let authMethod = 'none';

function logConfig() {
    const projectId = process.env.HUAWEI_PROJECT_ID || '未设置';
    const privateKeyFile = process.env.HUAWEI_PRIVATE_KEY_FILE || '未设置';
    const privateExists = privateKeyFile !== '未设置' ? fs.existsSync(privateKeyFile) : false;

    console.log(`[华为推送配置]`);
    console.log(`  PROJECT_ID:         ${projectId}`);
    console.log(`  PRIVATE_KEY_FILE:   ${privateKeyFile} (存在:${privateExists})`);
    console.log(`  鉴权方式:           JWT (HarmonyOS 5+ 必须使用JWT)`);
    
    if (!privateExists) {
        console.error(`[华为推送配置] ❌ 错误: private.json 文件不存在或路径未配置`);
        console.error(`[华为推送配置] 请设置环境变量 HUAWEI_PRIVATE_KEY_FILE 指向 private.json 文件路径`);
    }
}

logConfig();

function isValidPushToken(token) {
    if (!token || typeof token !== 'string') return false;
    const trimmed = token.trim();
    
    if (/^(test_|mock_|fake_|push_|dev_)/i.test(trimmed)) return false;
    if (/^test_tok/i.test(trimmed)) return false;
    if (/^AAA[A-Za-z0-9+/=_-]{10,}$/.test(trimmed)) return true;
    if (trimmed.length < 10) return false;
    
    return /^[A-Za-z0-9+/=_-]+$/.test(trimmed);
}

function validateToken(token) {
    if (!isValidPushToken(token)) {
        console.warn(`[Token校验] 无效Token 长度=${token ? token.length : 0} 值=${token ? token.substring(0, 20) : '空'}`);
        return false;
    }
    return true;
}

async function getAccessToken() {
    try {
        const privateKeyPath = process.env.HUAWEI_PRIVATE_KEY_FILE;
        if (!privateKeyPath) {
            console.error('[JWT鉴权] ❌ 未配置 HUAWEI_PRIVATE_KEY_FILE 环境变量');
            console.error('[JWT鉴权] 请在 .env 文件中设置: HUAWEI_PRIVATE_KEY_FILE=/path/to/private.json');
            return null;
        }
        
        if (!fs.existsSync(privateKeyPath)) {
            console.error('[JWT鉴权] ❌ private.json 文件不存在:', privateKeyPath);
            return null;
        }

        const privateData = JSON.parse(fs.readFileSync(privateKeyPath, 'utf8'));
        
        if (!privateData.private_key || !privateData.sub_account || !privateData.key_id) {
            console.error('[JWT鉴权] ❌ private.json 缺少必要字段 (private_key/sub_account/key_id)');
            return null;
        }

        const now = Math.floor(Date.now() / 1000);
        if (cachedToken && tokenExpireTime > now && authMethod === 'JWT(PS256)') {
            console.log('[JWT鉴权] 使用缓存的Token');
            return cachedToken;
        }

        console.log(`[JWT鉴权] 生成JWT: project_id=${privateData.project_id} sub_account=${privateData.sub_account} key_id=${privateData.key_id}`);

        const header = { alg: 'PS256', kid: privateData.key_id, typ: 'JWT' };
        const payload = {
            iss: privateData.sub_account,
            aud: 'https://oauth-login.cloud.huawei.com/oauth2/v3/token',
            iat: now,
            exp: now + 3600
        };

        const privateKeyRaw = privateData.private_key;
        const PRIVATE_KEY = privateKeyRaw.replace(/\\n/g, '\n');
        
        console.log(`[JWT鉴权] 私钥长度: ${PRIVATE_KEY.length} 字符`);

        let signedJwt;
        try {
            signedJwt = jwt.sign(payload, PRIVATE_KEY, { algorithm: 'PS256', header });
            console.log(`[JWT鉴权] ✅ JWT生成成功, 长度=${signedJwt.length}`);
        } catch (signErr) {
            console.error(`[JWT鉴权] ❌ JWT签名失败: ${signErr.message}`);
            return null;
        }

        cachedToken = signedJwt;
        tokenExpireTime = now + 3600 - 300;
        authMethod = 'JWT(PS256)';
        
        console.log(`[JWT鉴权] ✅ JWT Token已生成，有效期3600秒`);
        return cachedToken;
    } catch (err) {
        console.error(`[JWT鉴权] ❌ 异常: ${err.message}`);
        return null;
    }
}

const MESSAGE_CATEGORY = {
    IM: 'IM',
    VOIP: 'VOIP',
    MISS_CALL: 'MISS_CALL',
    SUBSCRIPTION: 'SUBSCRIPTION',
    TRAVEL: 'TRAVEL',
    HEALTH: 'HEALTH',
    WORK: 'WORK',
    ACCOUNT: 'ACCOUNT',
    EXPRESS: 'EXPRESS',
    FINANCE: 'FINANCE',
    DEVICE_REMINDER: 'DEVICE_REMINDER',
    MAIL: 'MAIL',
    MARKETING: 'MARKETING'
};

let selfCategoryEnabled = process.env.PUSH_SELF_CATEGORY === 'false' ? false : null;

function getCategoryByType(type) {
    if (selfCategoryEnabled === false) {
        return MESSAGE_CATEGORY.MARKETING;
    }
    const categoryMap = {
        new_grade: MESSAGE_CATEGORY.WORK,
        new_exam: MESSAGE_CATEGORY.WORK,
        exam_reminder: MESSAGE_CATEGORY.WORK,
        course_change: MESSAGE_CATEGORY.WORK,
        course_reminder: MESSAGE_CATEGORY.WORK,
        electricity_reminder: MESSAGE_CATEGORY.DEVICE_REMINDER,
        electricity_low: MESSAGE_CATEGORY.DEVICE_REMINDER,
        announcement: MESSAGE_CATEGORY.MARKETING
    };
    return categoryMap[type] || MESSAGE_CATEGORY.WORK;
}

async function sendPushNotification(pushToken, title, content, type, extraData, options = {}) {
    try {
        if (process.env.MOCK_MODE === 'true') {
            console.log(`[模拟推送] ${type || ''}: ${title}`);
            return { success: true, mock: true };
        }

        if (!pushToken) {
            return { success: false, message: '无推送Token' };
        }

        if (!validateToken(pushToken)) {
            return { success: false, message: '无效的推送Token（格式不合法）' };
        }

        const accessToken = await getAccessToken();
        if (!accessToken) {
            return { success: false, message: `华为推送服务不可用 (authMethod=${authMethod})` };
        }

        const projectId = process.env.HUAWEI_PROJECT_ID;
        if (!projectId) {
            return { success: false, message: '未配置HUAWEI_PROJECT_ID' };
        }

        const url = `https://push-api.cloud.huawei.com/v3/${projectId}/messages:send`;

        let category = options.category !== undefined ? options.category : getCategoryByType(type);
        const actionType = options.actionType !== undefined ? options.actionType : 0;
        const foregroundShow = options.foregroundShow !== undefined ? options.foregroundShow : true;
        const testMessage = options.testMessage || false;

        console.log(`[华为推送] URL=${url} | auth=${authMethod} | type=${type || '-'} | category=${category} | Token前缀=${pushToken.substring(0, 8)}...`);

        const buildPayload = (useCategory) => {
            const notification = {
                category: useCategory,
                title: title,
                body: content,
                clickAction: {
                    actionType: actionType,
                    data: extraData || {}
                },
                foregroundShow: foregroundShow,
                visibilityType: options.visibilityType !== undefined ? options.visibilityType : 1,
                badge: options.badge !== undefined ? options.badge : { addNum: 1 }
            };
            if (options.notifyId !== undefined) {
                notification.notifyId = options.notifyId;
            }
            if (options.sound !== undefined) {
                notification.sound = options.sound;
            }
            if (options.image !== undefined) {
                notification.image = options.image;
            }
            if (options.style !== undefined) {
                notification.style = options.style;
            }
            return {
                payload: { notification },
                target: { token: [pushToken] },
                pushOptions: { testMessage: testMessage, ttl: 86400 }
            };
        };

        const sendRequest = async (payload) => {
            return await axios.post(url, payload, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                    'push-type': '0'
                },
                timeout: 10000
            });
        };

        let payload = buildPayload(category);
        let res = await sendRequest(payload);
        let responseData = res.data;

        if (responseData.code === '80100003' && category !== MESSAGE_CATEGORY.MARKETING) {
            console.warn(`[华为推送] category="${category}" 无效，回退到 MARKETING...`);
            selfCategoryEnabled = false;
            category = MESSAGE_CATEGORY.MARKETING;
            payload = buildPayload(category);
            res = await sendRequest(payload);
            responseData = res.data;
        }

        if (responseData.code === '80000000') {
            console.log(`[华为推送] ✅ 发送成功 | auth=${authMethod} | type:${type || '-'} | category:${category} | 标题:${title}`);
            return { success: true, requestId: responseData.request_id, authMethod, category };
        } else {
            const errorCode = responseData.code;
            const errorMsg = responseData.msg;

            console.error(`[华为推送] ❌ 发送失败 code=${errorCode} msg=${errorMsg} auth=${authMethod}`);
            
            if (errorCode === '80300002') {
                console.error(`[华为推送诊断] ⚠️ 80300002 = 无权限。auth=${authMethod}`);
                console.error(`[华为推送诊断]   HarmonyOS 5+ 必须使用JWT认证，不支持OAuth`);
                console.error(`[华为推送诊断]   请检查 private.json 配置是否正确`);
            }
            if (errorCode === '80100003') {
                console.error(`[华为推送诊断] ⚠️ 80100003 = category无效或未申请自分类权益`);
                console.error(`[华为推送诊断]   有效值: IM, VOIP, MISS_CALL, SUBSCRIPTION, TRAVEL, HEALTH, WORK, ACCOUNT, EXPRESS, FINANCE, DEVICE_REMINDER, MAIL, MARKETING`);
                console.error(`[华为推送诊断]   需在AppGallery Connect中申请"通知消息自分类权益"才能使用非MARKETING类别`);
                console.error(`[华为推送诊断]   未申请权益时，所有消息默认为MARKETING(静默通知，仅通知中心展示)`);
            }
            if (errorCode === '80200001') {
                console.error(`[华为推送诊断] ⚠️ 80200001 = 认证失败`);
                console.error(`[华为推送诊断]   HarmonyOS 5+ 必须使用JWT认证，请检查:`);
                console.error(`[华为推送诊断]   1. private.json 文件是否存在且格式正确`);
                console.error(`[华为推送诊断]   2. PROJECT_ID 是否与 private.json 中的 project_id 一致`);
            }

            return { success: false, message: errorMsg || `错误码: ${errorCode}`, code: errorCode, authMethod };
        }

    } catch (err) {
        const errMsg = err.message || '';
        const statusCode = err.response?.status;
        const errorBody = err.response ? JSON.stringify(err.response.data).substring(0, 300) : '';

        console.error(`[华为推送] 异常 HTTP=${statusCode} | ${errMsg} | auth=${authMethod} | 响应:${errorBody}`);
        return { success: false, message: errMsg, httpStatus: statusCode, authMethod };
    }
}

async function sendBackgroundMessage(pushToken, extraData, options = {}) {
    try {
        if (process.env.MOCK_MODE === 'true') {
            console.log(`[后台消息] 模拟模式 Token:${pushToken ? pushToken.substring(0, 20) : '空'}...`);
            return { success: true, mock: true };
        }

        if (!pushToken) {
            return { success: false, message: '无推送Token' };
        }

        if (!validateToken(pushToken)) {
            return { success: false, message: '无效的推送Token（格式不合法）' };
        }

        const accessToken = await getAccessToken();
        if (!accessToken) {
            return { success: false, message: '华为推送服务不可用' };
        }

        const projectId = process.env.HUAWEI_PROJECT_ID;
        if (!projectId) {
            return { success: false, message: '未配置HUAWEI_PROJECT_ID' };
        }

        const url = `https://push-api.cloud.huawei.com/v3/${projectId}/messages:send`;

        const payload = {
            payload: {
                extraData: typeof extraData === 'string' ? extraData : JSON.stringify(extraData || {}),
                proxyData: options.enableProxy ? 'ENABLE' : undefined
            },
            target: {
                token: [pushToken]
            }
        };

        Object.keys(payload.payload).forEach(key => {
            if (payload.payload[key] === undefined) delete payload.payload[key];
        });

        const res = await axios.post(url, payload, {
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json',
                'push-type': '6'
            },
            timeout: 10000
        });

        const responseData = res.data;

        if (responseData.code === '80000000') {
            console.log(`[后台消息] ✅ 发送成功 | auth=${authMethod} | RequestId:${responseData.request_id}`);
            return { success: true, requestId: responseData.request_id, authMethod };
        } else {
            const errorCode = responseData.code;
            const errorMsg = responseData.msg;

            console.error(`[后台消息] ❌ 发送失败 code=${errorCode} msg=${errorMsg} auth=${authMethod}`);

            return { success: false, message: errorMsg || `错误码: ${errorCode}`, code: errorCode, authMethod };
        }

    } catch (err) {
        const errMsg = err.message || '';
        const statusCode = err.response?.status;
        const errorBody = err.response ? JSON.stringify(err.response.data).substring(0, 300) : '';

        console.error(`[后台消息] 异常 HTTP=${statusCode} | ${errMsg} | auth=${authMethod} | 响应:${errorBody}`);
        return { success: false, message: errMsg, httpStatus: statusCode, authMethod };
    }
}

async function notifyNewGrade(studentId, gradeInfo) {
    const { UserPushToken } = require('../db/models');

    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });

        if (!userToken) {
            console.log(`[成绩通知] 用户 ${studentId} 未注册推送Token`);
            return { success: false, message: '未注册推送' };
        }

        return await sendPushNotification(
            userToken.pushToken,
            '新成绩发布',
            `${gradeInfo.courseName || '课程'}: ${gradeInfo.score || '未知'}分 (${gradeInfo.credit || '?'}学分)`,
            'new_grade',
            { studentId, courseName: gradeInfo.courseName, score: gradeInfo.score, semester: gradeInfo.semester },
            { visibilityType: 1, badge: { addNum: 1 } }
        );
    } catch (error) {
        console.error(`[成绩通知] 发送失败 (${studentId}):`, error.message);
        return { success: false, message: error.message };
    }
}

async function notifyNewExam(studentId, examInfo) {
    const { UserPushToken } = require('../db/models');

    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });

        if (!userToken) {
            console.log(`[考试通知] 用户 ${studentId} 未注册推送Token`);
            return { success: false, message: '未注册推送' };
        }

        return await sendPushNotification(
            userToken.pushToken,
            '新考试安排',
            `${examInfo.courseName || '考试'}: ${examInfo.examTime || '时间待定'} @ ${examInfo.location || '地点待定'}`,
            'new_exam',
            { studentId, courseName: examInfo.courseName, examTime: examInfo.examTime, location: examInfo.location },
            { visibilityType: 1, badge: { addNum: 1 } }
        );
    } catch (error) {
        console.error(`[考试通知] 发送失败 (${studentId}):`, error.message);
        return { success: false, message: error.message };
    }
}

async function notifyExamReminder(studentId, examInfo) {
    const { UserPushToken } = require('../db/models');

    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });

        if (!userToken) {
            console.log(`[考试提醒] 用户 ${studentId} 未注册推送Token`);
            return { success: false, message: '未注册推送' };
        }

        return await sendPushNotification(
            userToken.pushToken,
            '⏰ 考试提醒',
            `${examInfo.courseName || '考试'} 将在${examInfo.reminderTime || '24小时后'}开始\n时间: ${examInfo.examTime || '-'}\n地点: ${examInfo.location || '-'}`,
            'exam_reminder',
            { studentId, courseName: examInfo.courseName, examTime: examInfo.examTime, location: examInfo.location, reminderType: examInfo.reminderTime },
            { visibilityType: 1, badge: { addNum: 1 } }
        );
    } catch (error) {
        console.error(`[考试提醒] 发送失败 (${studentId}):`, error.message);
        return { success: false, message: error.message };
    }
}

async function notifyCourseChange(studentId, changeInfo) {
    const { UserPushToken } = require('../db/models');

    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });

        if (!userToken) {
            console.log(`[课变通知] 用户 ${studentId} 未注册推送Token`);
            return { success: false, message: '未注册推送' };
        }

        const typeLabels = {
            new: '🆕 新增课程',
            location_change: '📍 教室变更',
            cancelled: '❌ 课程取消'
        };

        return await sendPushNotification(
            userToken.pushToken,
            typeLabels[changeInfo.type] || '课表变动',
            changeInfo.message || '课表有变动',
            'course_change',
            { studentId, type: changeInfo.type, courseName: changeInfo.courseName, message: changeInfo.message },
            { visibilityType: 1, badge: { addNum: 1 } }
        );
    } catch (error) {
        console.error(`[课变通知] 发送失败 (${studentId}):`, error.message);
        return { success: false, message: error.message };
    }
}

async function notifyAnnouncement(title, keyword, url) {
    try {
        const allTokensResult = await require('../db/models').UserPushToken.findAll({
            where: { isActive: true }
        });

        if (!allTokensResult || allTokensResult.length === 0) {
            console.log('[公告通知] 无活跃用户');
            return { success: false, message: '无活跃用户' };
        }

        let successCount = 0;
        let failCount = 0;
        const errors = [];

        for (const user of allTokensResult) {
            const result = await sendPushNotification(
                user.pushToken,
                `📢 公告: ${keyword || '重要'}`,
                `${title}${url ? '\n点击查看详情' : ''}`,
                'announcement',
                { keyword, url, title },
                { visibilityType: 1, badge: { addNum: 1 } }
            );

            if (result.success) {
                successCount++;
            } else {
                failCount++;
                errors.push({
                    studentId: user.studentId,
                    message: result.message,
                    code: result.code
                });
                console.error(`[公告通知] 推送失败 studentId=${user.studentId} code=${result.code} msg=${result.message}`);
            }
        }

        console.log(`[公告通知] 推送完成 成功:${successCount} 失败:${failCount}`);
        if (errors.length > 0 && errors.length <= 3) {
            console.log(`[公告通知] 失败详情:`, JSON.stringify(errors, null, 2));
        }
        return { success: true, successCount, failCount, errors };
    } catch (error) {
        console.error('[公告通知] 异常:', error.message);
        return { success: false, message: error.message };
    }
}

async function notifyCourseReminder(studentId, reminderInfo) {
    const { UserPushToken } = require('../db/models');

    try {
        const userToken = await UserPushToken.findOne({
            where: { studentId, isActive: true }
        });

        if (!userToken) {
            console.log(`[课程提醒] 用户 ${studentId} 未注册推送Token`);
            return { success: false, message: '未注册推送' };
        }

        const typeLabels = {
            before_class: '📚 课前提醒',
            tomorrow: '📅 明日课程',
            morning: '☀️ 今日课程'
        };

        return await sendPushNotification(
            userToken.pushToken,
            typeLabels[reminderInfo.type] || '课程提醒',
            reminderInfo.message || '您有课程即将开始',
            'course_reminder',
            {
                studentId,
                type: reminderInfo.type,
                courseName: reminderInfo.courseName,
                location: reminderInfo.location,
                startTime: reminderInfo.startTime
            },
            { visibilityType: 1, badge: { addNum: 1 } }
        );
    } catch (error) {
        console.error(`[课程提醒] 推送失败 (${studentId}):`, error.message);
        return { success: false, message: error.message };
    }
}

module.exports = {
    sendPushNotification,
    sendBackgroundMessage,
    notifyNewGrade,
    notifyNewExam,
    notifyExamReminder,
    notifyCourseChange,
    notifyCourseReminder,
    notifyAnnouncement,
    getAccessToken,
    validateToken,
    isValidPushToken,
    MESSAGE_CATEGORY,
    getCategoryByType
};
