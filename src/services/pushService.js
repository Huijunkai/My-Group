const axios = require('axios');

const HUAWEI_PUSH_API = 'https://push-api.cloud.huawei.com/v3';
const PROJECT_ID = process.env.HUAWEI_PROJECT_ID || 'YOUR_PROJECT_ID';

let accessToken = null;
let tokenExpireTime = 0;

async function getAccessToken() {
    if (accessToken && Date.now() < tokenExpireTime) {
        return accessToken;
    }

    const clientId = process.env.HUAWEI_CLIENT_ID;
    const clientSecret = process.env.HUAWEI_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
        console.warn('PushService: Huawei credentials not configured, push notifications disabled');
        return null;
    }

    try {
        const params = new URLSearchParams();
        params.append('grant_type', 'client_credentials');
        params.append('client_id', clientId);
        params.append('client_secret', clientSecret);
        
        console.log('PushService: Requesting access token with clientId:', clientId.substring(0, 10) + '...');
        
        const response = await axios.post('https://oauth-login.cloud.huawei.com/oauth2/v3/token', params, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        accessToken = response.data.access_token;
        tokenExpireTime = Date.now() + (response.data.expires_in - 300) * 1000;

        console.log('PushService: Access token obtained successfully');
        return accessToken;
    } catch (error) {
        console.error('PushService: Failed to get access token:', error.message);
        if (error.response) {
            console.error('PushService: Error response data:', error.response.data);
            console.error('PushService: Error response status:', error.response.status);
        }
        return null;
    }
}

async function sendPushNotification(token, title, content, type, extraData = {}) {
    const accessToken = await getAccessToken();
    if (!accessToken) {
        return { success: false, message: 'Failed to get access token' };
    }

    try {
        const message = {
            validate_only: false,
            message: {
                android: {
                    notification: {
                        title: title,
                        body: content,
                        click_action: {
                            type: 3
                        }
                    }
                },
                token: [token],
                data: JSON.stringify({
                    type: type,
                    ...extraData
                })
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        console.log(`PushService: Push sent successfully, type=${type}, title=${title}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send push notification:', error.message);
        return { success: false, message: error.message };
    }
}

async function sendPushToUser(anonymousId, title, content, type, extraData = {}) {
    const accessToken = await getAccessToken();
    if (!accessToken) {
        return { success: false, message: 'Failed to get access token' };
    }

    try {
        const message = {
            validate_only: false,
            message: {
                android: {
                    notification: {
                        title: title,
                        body: content,
                        click_action: {
                            type: 3
                        }
                    }
                },
                topic: `user_${anonymousId}`,
                data: JSON.stringify({
                    type: type,
                    ...extraData
                })
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        console.log(`PushService: Push sent to user ${anonymousId}, type=${type}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send push to user:', error.message);
        return { success: false, message: error.message };
    }
}

async function notifyCourseChange(studentId, changeInfo) {
    const anonymousId = generateAnonymousId(studentId);
    const title = '课程变动通知';
    const content = changeInfo.message || '您的课表有新的变动，请及时查看';
    
    return await sendPushToUser(anonymousId, title, content, 'course_change', {
        changeType: changeInfo.type,
        courseName: changeInfo.courseName
    });
}

async function notifyNewGrade(studentId, gradeInfo) {
    const anonymousId = generateAnonymousId(studentId);
    const title = '新成绩发布';
    const content = `${gradeInfo.courseName}: ${gradeInfo.score}分`;
    
    return await sendPushToUser(anonymousId, title, content, 'new_grade', {
        courseName: gradeInfo.courseName,
        score: gradeInfo.score,
        credit: gradeInfo.credit,
        semester: gradeInfo.semester
    });
}

async function notifyNewExam(studentId, examInfo) {
    const anonymousId = generateAnonymousId(studentId);
    const title = '新考试安排';
    const content = `${examInfo.courseName} - ${examInfo.examTime}`;
    
    return await sendPushToUser(anonymousId, title, content, 'new_exam', {
        courseName: examInfo.courseName,
        examTime: examInfo.examTime,
        location: examInfo.location
    });
}

async function notifyExamReminder(studentId, examInfo) {
    const anonymousId = generateAnonymousId(studentId);
    const title = '考试提醒';
    const content = `${examInfo.courseName} 将在 ${examInfo.reminderTime} 开始`;
    
    return await sendPushToUser(anonymousId, title, content, 'exam_reminder', {
        courseName: examInfo.courseName,
        examTime: examInfo.examTime,
        location: examInfo.location
    });
}

async function notifyElectricityLow(studentId, electricityInfo) {
    const anonymousId = generateAnonymousId(studentId);
    const title = '电费提醒';
    const content = `您的宿舍电费余额已低于 ${electricityInfo.threshold} 元，当前余额为 ${electricityInfo.balance.toFixed(2)} 元，请及时充值。`;
    
    return await sendPushToUser(anonymousId, title, content, 'electricity_reminder', {
        balance: electricityInfo.balance,
        threshold: electricityInfo.threshold
    });
}

async function notifyAnnouncement(title, keyword, url) {
    const keywordLabels = {
        '重修': '重修通知',
        '补考': '补考通知',
        '体质健康测试': '体质健康测试通知',
        '选课': '选课通知',
        '补修': '补修通知',
        '免修': '免修通知'
    };
    
    const label = keywordLabels[keyword] || '教务通知';
    const content = `【${label}】${title}`;
    
    return await sendPushToAllUsers(label, content, 'announcement', {
        title: title,
        keyword: keyword,
        url: url
    });
}

async function sendPushToAllUsers(title, content, type, extraData = {}) {
    const accessToken = await getAccessToken();
    if (!accessToken) {
        return { success: false, message: 'Failed to get access token' };
    }

    try {
        const message = {
            validate_only: false,
            message: {
                android: {
                    notification: {
                        title: title,
                        body: content,
                        click_action: {
                            type: 3
                        }
                    }
                },
                topic: 'announcements',
                data: JSON.stringify({
                    type: type,
                    ...extraData
                })
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        console.log(`PushService: Broadcast push sent, type=${type}, title=${title}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send broadcast push:', error.message);
        return { success: false, message: error.message };
    }
}

function generateAnonymousId(studentId) {
    let hash = 0;
    for (let i = 0; i < studentId.length; i++) {
        const char = studentId.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return `nnlg_${Math.abs(hash).toString(16)}`;
}

module.exports = {
    sendPushNotification,
    sendPushToUser,
    sendPushToAllUsers,
    notifyCourseChange,
    notifyNewGrade,
    notifyNewExam,
    notifyExamReminder,
    notifyElectricityLow,
    notifyAnnouncement,
    generateAnonymousId
};
