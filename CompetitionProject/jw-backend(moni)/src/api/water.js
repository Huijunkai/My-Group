const axios = require('axios');

const WATER_BASE_URL = 'https://server.happy-ti.com';
const WX_PAGE_BASE = 'http://wx.happy-ti.com';

const WATER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Origin': WX_PAGE_BASE,
    'Referer': `${WX_PAGE_BASE}/`
};

function parseScanUrl(scanUrl) {
    try {
        const url = new URL(scanUrl);
        const params = new URLSearchParams(url.search);
        
        const openid = params.get('openid');
        const deviceid = params.get('deviceid');
        const app = params.get('app') || 'WECHAT';
        const token = params.get('token') || '';
        const ch = params.get('ch') || '';

        if (!openid || !deviceid) {
            return { success: false, message: '缺少必要参数: openid 或 deviceid' };
        }

        return {
            success: true,
            data: { openid, deviceid, app, token, ch }
        };
    } catch (error) {
        return { success: false, message: 'URL 解析失败: ' + error.message };
    }
}

async function initWaterDevice(openid, deviceid, app = 'WECHAT') {
    try {
        const url = `${WATER_BASE_URL}/index.php`;
        const params = {
            r: 'api/wxpay/v1/scanqrcode/initv0',
            deviceid: deviceid,
            openid: openid,
            app: app
        };

        console.log(`打水系统: 正在初始化设备 ${deviceid}`);
        
        const response = await axios.get(url, {
            params: params,
            headers: WATER_HEADERS
        });

        const result = response.data;
        
        if (result.code === 0) {
            console.log(`打水系统: 设备初始化成功 - 位置: ${result.data?.location}, 用户: ${result.data?.userid}`);
            return {
                success: true,
                data: result.data
            };
        } else {
            return {
                success: false,
                message: result.msg || '初始化失败',
                code: result.code
            };
        }
    } catch (error) {
        console.error('打水系统初始化失败:', error.message);
        return {
            success: false,
            message: '请求失败: ' + error.message
        };
    }
}

async function getWaterBalance(openid, saler, app = 'WECHAT') {
    try {
        const url = `${WATER_BASE_URL}/index.php`;
        const params = {
            r: 'api/server/v1/cards/getcards',
            openid: openid,
            saler: saler || '',
            app: app
        };

        console.log(`打水系统: 正在获取用户余额`);
        
        const response = await axios.get(url, {
            params: params,
            headers: WATER_HEADERS
        });

        const result = response.data;
        
        if (result.code === 0 && result.data && result.data.length > 0) {
            const cardInfo = result.data[0];
            const balance = cardInfo.cash || cardInfo.totalvalue || '0';
            const cardNo = cardInfo.number || '';
            const ownerName = cardInfo.owner_name || '';
            
            console.log(`打水系统: 获取余额成功 - 余额: ${balance}元, 会员: ${ownerName}`);
            return {
                success: true,
                data: {
                    balance: String(balance),
                    cardNo: cardNo,
                    ownerName: ownerName,
                    userid: cardInfo.owner || ''
                }
            };
        } else {
            return {
                success: false,
                message: result.msg || '获取余额失败'
            };
        }
    } catch (error) {
        console.error('打水系统获取余额失败:', error.message);
        return {
            success: false,
            message: '请求失败: ' + error.message
        };
    }
}

async function scanWaterQrcode(scanUrl) {
    const parseResult = parseScanUrl(scanUrl);
    
    if (!parseResult.success) {
        return parseResult;
    }

    const { openid, deviceid, app } = parseResult.data;
    return await initWaterDevice(openid, deviceid, app);
}

async function bindWaterAccount(scanUrl) {
    const parseResult = parseScanUrl(scanUrl);
    
    if (!parseResult.success) {
        return parseResult;
    }

    const { openid, deviceid, app } = parseResult.data;
    const result = await initWaterDevice(openid, deviceid, app);
    
    if (result.success && result.data) {
        const balanceResult = await getWaterBalance(openid, result.data.saler, app);

        return {
            success: true,
            data: {
                userid: result.data.userid,
                location: result.data.location,
                saler: result.data.saler,
                openid: openid,
                deviceid: deviceid,
                balance: balanceResult.success ? balanceResult.data.balance : null,
                cardNo: balanceResult.success ? balanceResult.data.cardNo : null,
                ownerName: balanceResult.success ? balanceResult.data.ownerName : null
            }
        };
    }
    
    return result;
}

module.exports = {
    parseScanUrl,
    initWaterDevice,
    scanWaterQrcode,
    getWaterBalance,
    bindWaterAccount
};
