const axios = require('axios');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const HUAWEI_PUSH_API = 'https://push-api.cloud.huawei.com/v3';
const PROJECT_ID = process.env.HUAWEI_PROJECT_ID || 'YOUR_PROJECT_ID';

let jwtConfig = null;
let accessToken = null;
let tokenExpireTime = 0;

function loadJwtConfig() {
    if (jwtConfig) return jwtConfig;
    
    const keyFile = process.env.HUAWEI_KEY_FILE;
    
    if (keyFile && fs.existsSync(keyFile)) {
        try {
            const content = fs.readFileSync(keyFile, 'utf8');
            jwtConfig = JSON.parse(content);
            console.log('PushService: Loaded JWT config from file:', keyFile);
            return jwtConfig;
        } catch (e) {
            console.error('PushService: Failed to read key file:', e.message);
        }
    }
    
    const keyId = process.env.HUAWEI_KEY_ID;
    const subAccount = process.env.HUAWEI_SUB_ACCOUNT;
    const privateKey = process.env.HUAWEI_PRIVATE_KEY;
    
    if (keyId && subAccount && privateKey) {
        jwtConfig = {
            key_id: keyId,
            sub_account: subAccount,
            private_key: privateKey
        };
        console.log('PushService: Loaded JWT config from env variables');
        return jwtConfig;
    }
    
    return null;
}

function base64UrlEncode(str) {
    return Buffer.from(str)
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=/g, '');
}

function generateJwt() {
    const config = loadJwtConfig();
    
    console.log('PushService: Checking JWT config...');
    console.log('  Config loaded:', !!config);
    
    if (!config || !config.key_id || !config.sub_account || !config.private_key) {
        console.error('PushService: Missing JWT configuration');
        console.error('  key_id:', !!config?.key_id);
        console.error('  sub_account:', !!config?.sub_account);
        console.error('  private_key:', !!config?.private_key, config?.private_key?.length || 0);
        return null;
    }

    const now = Math.floor(Date.now() / 1000);
    const exp = now + 3600;

    const header = {
        kid: config.key_id,
        typ: 'JWT',
        alg: 'PS256'
    };

    const payload = {
        aud: 'https://oauth-login.cloud.huawei.com/oauth2/v3/token',
        iss: config.sub_account,
        exp: exp,
        iat: now
    };

    try {
        const encodedHeader = base64UrlEncode(JSON.stringify(header));
        const encodedPayload = base64UrlEncode(JSON.stringify(payload));
        const signingInput = `${encodedHeader}.${encodedPayload}`;

        let privateKeyFormatted = config.private_key;
        
        if (!privateKeyFormatted.includes('-----BEGIN')) {
            privateKeyFormatted = `-----BEGIN PRIVATE KEY-----\n${privateKeyFormatted}\n-----END PRIVATE KEY-----`;
        }
        
        privateKeyFormatted = privateKeyFormatted.replace(/\\n/g, '\n');

        console.log('PushService: Private key length:', privateKeyFormatted.length);
        console.log('PushService: Private key starts with:', privateKeyFormatted.substring(0, 30));

        const sign = crypto.createSign('RSA-SHA256');
        sign.update(signingInput);
        sign.end();

        const signature = sign.sign({
            key: privateKeyFormatted,
            padding: crypto.constants.RSA_PKCS1_PSS_PADDING,
            saltLength: crypto.constants.RSA_PSS_SALT_LEN_DIGEST
        }, 'base64')
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=/g, '');

        const jwt = `${signingInput}.${signature}`;
        console.log('PushService: JWT generated successfully, length:', jwt.length);
        return jwt;
    } catch (error) {
        console.error('PushService: JWT generation failed:', error.message);
        return null;
    }
}

async function getAccessToken() {
    if (accessToken && Date.now() < tokenExpireTime) {
        return accessToken;
    }

    try {
        console.log('PushService: Getting access token...');
        
        const jwt = generateJwt();
        if (!jwt) {
            throw new Error('Failed to generate JWT');
        }
        
        console.log('PushService: Requesting access token with JWT...');

        const params = new URLSearchParams();
        params.append('grant_type', 'urn:ietf:params:oauth:grant-type:jwt-bearer');
        params.append('assertion', jwt);

        const response = await axios.post('https://oauth-login.cloud.huawei.com/oauth2/v3/token', params, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            timeout: 10000
        });

        accessToken = response.data.access_token;
        tokenExpireTime = Date.now() + (response.data.expires_in - 300) * 1000;

        console.log('PushService: Access token obtained successfully');
        return accessToken;
    } catch (error) {
        console.error('PushService: Failed to get access token:', error.message);
        if (error.response) {
            console.error('PushService: Error response data:', JSON.stringify(error.response.data));
            console.error('PushService: Error response status:', error.response.status);
        }
        return null;
    }
}

async function sendPushNotification(token, title, content, type, extraData = {}) {
    try {
        console.log('PushService: Sending push notification...');
        
        const authJwt = generateJwt();
        if (!authJwt) {
            return { success: false, message: 'Failed to generate JWT for push' };
        }

        console.log('PushService: Sending to token:', token.substring(0, 20) + '...');
        
        const message = {
            payload: {
                notification: {
                    category: 'MARKETING',
                    title: title,
                    body: content,
                    clickAction: {
                        actionType: 0,
                        data: {
                            type: type,
                            ...extraData
                        }
                    },
                    foregroundShow: true
                }
            },
            target: {
                token: [token]
            },
            pushOptions: {
                testMessage: true,
                ttl: 86400
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${authJwt}`,
                    'Content-Type': 'application/json',
                    'push-type': '0'
                },
                timeout: 15000
            }
        );

        console.log(`PushService: Push sent successfully, type=${type}, title=${title}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send push notification:', error.message);
        if (error.response) {
            console.error('PushService: Error response data:', JSON.stringify(error.response.data));
        }
        return { success: false, message: error.message };
    }
}

async function sendPushToUser(anonymousId, title, content, type, extraData = {}) {
    try {
        const authJwt = generateJwt();
        if (!authJwt) {
            return { success: false, message: 'Failed to generate JWT' };
        }

        const message = {
            payload: {
                notification: {
                    category: 'MARKETING',
                    title: title,
                    body: content,
                    clickAction: {
                        actionType: 0,
                        data: {
                            type: type,
                            ...extraData
                        }
                    },
                    foregroundShow: true
                }
            },
            target: {
                topic: `user_${anonymousId}`
            },
            pushOptions: {
                testMessage: true,
                ttl: 86400
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${authJwt}`,
                    'Content-Type': 'application/json',
                    'push-type': '0'
                },
                timeout: 15000
            }
        );

        console.log(`PushService: Push sent to user ${anonymousId}, type=${type}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send push to user:', error.message);
        if (error.response) {
            console.error('PushService: Error response data:', JSON.stringify(error.response.data));
        }
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
    const balance = typeof electricityInfo.balance === 'number' ? electricityInfo.balance : parseFloat(electricityInfo.balance) || 0;
    const content = `您的宿舍电费余额已低于 ${electricityInfo.threshold} 元，当前余额为 ${balance.toFixed(2)} 元，请及时充值。`;
    
    return await sendPushToUser(anonymousId, title, content, 'electricity_reminder', {
        balance: balance,
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
    try {
        const authJwt = generateJwt();
        if (!authJwt) {
            return { success: false, message: 'Failed to generate JWT' };
        }

        const message = {
            payload: {
                notification: {
                    category: 'MARKETING',
                    title: title,
                    body: content,
                    clickAction: {
                        actionType: 0,
                        data: {
                            type: type,
                            ...extraData
                        }
                    },
                    foregroundShow: true
                }
            },
            target: {
                topic: 'announcements'
            },
            pushOptions: {
                testMessage: true,
                ttl: 86400
            }
        };

        const response = await axios.post(
            `${HUAWEI_PUSH_API}/${PROJECT_ID}/messages:send`,
            message,
            {
                headers: {
                    'Authorization': `Bearer ${authJwt}`,
                    'Content-Type': 'application/json',
                    'push-type': '0'
                },
                timeout: 15000
            }
        );

        console.log(`PushService: Broadcast push sent, type=${type}, title=${title}`);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('PushService: Failed to send broadcast push:', error.message);
        if (error.response) {
            console.error('PushService: Error response data:', JSON.stringify(error.response.data));
        }
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
