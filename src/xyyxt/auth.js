const { BASE_URL, AUTH_URL, ELEC_URL, ELEC_URL_NANNING, ELEC_URL_GUILIN, BASIC_AUTH, DEFAULT_HEADERS, CAMPUS_CONFIG } = require('./constants');
const axios = require('axios');
const FormData = require('form-data');

function getElecUrl(areaId = '') {
    if (areaId && CAMPUS_CONFIG[areaId]) {
        return CAMPUS_CONFIG[areaId].elecUrl;
    }
    return ELEC_URL;
}

const GUILIN_ELEC_URL_1 = 'http://221.7.150.22:10005';
const GUILIN_ELEC_URL_2 = 'http://221.7.150.20:10004';
const GUILIN_ELEC_URL_3 = 'http://202.103.236.36:10001';

const GUILIN_ROOM_URL_1 = 'http://221.7.150.22:10005';
const GUILIN_ROOM_URL_2 = 'http://221.7.150.20:10004';
const GUILIN_ROOM_URL_3 = 'http://202.103.236.36:10001';

const GUILIN_BUILDING_URL_MAP = {
    '4320': GUILIN_ROOM_URL_1,
    '4509': GUILIN_ROOM_URL_1,
    '4722': GUILIN_ROOM_URL_1,
    '4812': GUILIN_ROOM_URL_1,
    '6436': GUILIN_ROOM_URL_1,
    '6819': GUILIN_ROOM_URL_1,
    'B101': GUILIN_ROOM_URL_2,
    'B102': GUILIN_ROOM_URL_2,
    'B8': GUILIN_ROOM_URL_2
};

const NANNING_ELEC_URL = 'http://202.103.236.36:10001';

const NANNING_BUILDING_URL_MAP = {
    '4320': NANNING_ELEC_URL,
    '4523': NANNING_ELEC_URL,
    '4722': NANNING_ELEC_URL,
    '5158': NANNING_ELEC_URL,
    '5623': NANNING_ELEC_URL,
    '6068': NANNING_ELEC_URL,
    '6267': NANNING_ELEC_URL,
    '6454': NANNING_ELEC_URL,
    '6899': NANNING_ELEC_URL,
    'B1': NANNING_ELEC_URL,
    'B2': NANNING_ELEC_URL,
    'B3': NANNING_ELEC_URL,
    'B4': NANNING_ELEC_URL,
    'B5': NANNING_ELEC_URL,
    'B6': NANNING_ELEC_URL,
    'B7': NANNING_ELEC_URL,
    'B8': NANNING_ELEC_URL,
    'B9': NANNING_ELEC_URL,
    'B10': NANNING_ELEC_URL,
    'B11': NANNING_ELEC_URL,
    'B12': NANNING_ELEC_URL,
    'B16': NANNING_ELEC_URL,
    'B19': NANNING_ELEC_URL,
    'B20': NANNING_ELEC_URL,
    'B21': NANNING_ELEC_URL,
    'B22': NANNING_ELEC_URL,
    'B23': NANNING_ELEC_URL,
    'B24': NANNING_ELEC_URL,
    'B25': NANNING_ELEC_URL
};

const GUILIN_ELEC_PREFIX_MAP = {
    'H432': GUILIN_ELEC_URL_1,
    'H450': GUILIN_ELEC_URL_1,
    'H472': GUILIN_ELEC_URL_1,
    'H481': GUILIN_ELEC_URL_1,
    'H643': GUILIN_ELEC_URL_1,
    'H681': GUILIN_ELEC_URL_1,
    'HB10': GUILIN_ELEC_URL_2,
    'HB8': GUILIN_ELEC_URL_2
};

function getElecUrlForRoom(areaId = '', roomId = '') {
    if (areaId === 'glxq' && roomId) {
        for (const [prefix, url] of Object.entries(GUILIN_ELEC_PREFIX_MAP)) {
            if (roomId.startsWith(prefix)) {
                return url;
            }
        }
    }
    return getElecUrl(areaId);
}

function getRoomUrl(areaId = '', buildingId = '') {
    if (areaId === 'glxq' && buildingId && GUILIN_BUILDING_URL_MAP[buildingId]) {
        return GUILIN_BUILDING_URL_MAP[buildingId];
    }
    if (areaId === 'nnxq' && buildingId && NANNING_BUILDING_URL_MAP[buildingId]) {
        return NANNING_BUILDING_URL_MAP[buildingId];
    }
    if (areaId && CAMPUS_CONFIG[areaId] && CAMPUS_CONFIG[areaId].roomUrl) {
        return CAMPUS_CONFIG[areaId].roomUrl;
    }
    return getElecUrl(areaId);
}

function formatCookies(cookies) {
    if (!cookies) return '';
    if (Array.isArray(cookies)) {
        return cookies.map(c => c.split(';')[0]).join('; ');
    }
    return cookies;
}

function createInstance(cookies = '', referer = '', authHeader = '') {
    const headers = { ...DEFAULT_HEADERS };
    if (cookies) {
        headers['Cookie'] = formatCookies(cookies);
    }
    if (referer) {
        headers['Referer'] = referer;
    }
    if (authHeader) {
        headers['Authorization'] = authHeader;
    }

    return axios.create({
        headers,
        withCredentials: true,
        timeout: 30000
    });
}

async function login(username, password) {
    try {
        const instance = createInstance('', '', BASIC_AUTH);
        
        const loginUrl = `${AUTH_URL}/authentication/form`;
        
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        formData.append('grant_type', 'password');
        
        const response = await instance.post(loginUrl, formData.toString(), {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data && response.data.access_token) {
            const cookies = response.headers['set-cookie'] || [];
            return {
                success: true,
                cookies: cookies,
                data: {
                    access_token: response.data.access_token,
                    refresh_token: response.data.refresh_token,
                    expires_in: response.data.expires_in,
                    schoolId: response.data.schoolId,
                    userId: response.data.userId
                }
            };
        } else {
            return {
                success: false,
                message: response.data?.message || response.data?.error_description || '登录失败，请检查账号密码'
            };
        }
    } catch (error) {
        console.error('校园一信通登录失败:', error.message);
        if (error.response) {
            console.error('响应状态:', error.response.status);
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return {
            success: false,
            message: error.response?.data?.message || error.response?.data?.error_description || error.message || '登录请求失败'
        };
    }
}

async function refreshToken(refreshToken) {
    try {
        const instance = createInstance('', '', BASIC_AUTH);
        
        const url = `${AUTH_URL}/authentication/refresh`;
        
        const formData = new URLSearchParams();
        formData.append('grant_type', 'refresh_token');
        formData.append('refresh_token', refreshToken);
        
        const response = await instance.post(url, formData.toString(), {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data && response.data.access_token) {
            return {
                success: true,
                data: response.data
            };
        } else {
            return {
                success: false,
                message: response.data?.message || '刷新Token失败'
            };
        }
    } catch (error) {
        console.error('刷新Token失败:', error.message);
        return {
            success: false,
            message: error.message || '刷新Token请求失败'
        };
    }
}

async function getUserInfo(accessToken) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${AUTH_URL}/user/info`);
        
        if (response.data && response.data.retCode === 200) {
            return response.data.data || null;
        }
        return null;
    } catch (error) {
        console.error('获取用户信息失败:', error.message);
        return null;
    }
}

async function getBalance(accessToken) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${BASE_URL}/bwgl_remoteservice/cardController/getAccountBln`);
        
        if (response.data && response.data.code === 200) {
            return response.data.data || response.data || null;
        }
        return response.data || null;
    } catch (error) {
        console.error('获取余额失败:', error.message);
        return null;
    }
}

async function getTransactions(accessToken, page = 1, size = 20) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${BASE_URL}/yxtapp/mobile/card/transactions`, {
            params: { page, size }
        });
        
        if (response.data && response.data.retCode === 200) {
            return response.data.data || [];
        }
        return [];
    } catch (error) {
        console.error('获取交易记录失败:', error.message);
        return [];
    }
}

async function getConsumptionRecords(accessToken, page = 1, size = 20, timeRange = '') {
    try {
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        if (!timeRange) {
            const now = new Date();
            const endDate = now.toISOString().split('T')[0];
            const startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            timeRange = `${startDate}~${endDate}`;
        }
        
        const url = `${BASE_URL}/bwgl_remoteservice/consumption/newConsumption?opr=2&current=${page}&size=${size}&time=${encodeURIComponent(timeRange)}&reason=`;
        
        const formData = new FormData();
        
        const response = await axios.post(url, formData, { headers });
        
        console.log('消费记录响应:', JSON.stringify(response.data));
        
        if (response.data && response.data.code === 200) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取消费记录失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getRechargeRecords(accessToken, page = 1, size = 20, timeRange = '') {
    try {
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        if (!timeRange) {
            const now = new Date();
            const endDate = now.toISOString().split('T')[0];
            const startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            timeRange = `${startDate}~${endDate}`;
        }
        
        const url = `${BASE_URL}/bwgl_remoteservice/consumption/newConsumption?opr=1&current=${page}&size=${size}&time=${encodeURIComponent(timeRange)}&reason=`;
        
        const formData = new FormData();
        
        const response = await axios.post(url, formData, { headers });
        
        console.log('充值记录响应:', JSON.stringify(response.data));
        
        if (response.data && response.data.code === 200) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取充值记录失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

const GUILIN_BUILDINGS = [
    { loudong_id: '4320', loudong_name: '9栋' },
    { loudong_id: '4509', loudong_name: '7栋' },
    { loudong_id: '4722', loudong_name: '12栋' },
    { loudong_id: '4812', loudong_name: '13栋' },
    { loudong_id: '6436', loudong_name: '14A栋' },
    { loudong_id: '6819', loudong_name: '14B栋' },
    { loudong_id: 'B101', loudong_name: '10A栋' },
    { loudong_id: 'B102', loudong_name: '10B栋' },
    { loudong_id: 'B8', loudong_name: '8栋' }
];

const NANNING_BUILDINGS = [
    { loudong_id: '4320', loudong_name: '15-1栋' },
    { loudong_id: '4523', loudong_name: '15-2栋' },
    { loudong_id: '4722', loudong_name: '13-1栋' },
    { loudong_id: '5158', loudong_name: '13-2栋' },
    { loudong_id: '5623', loudong_name: '17栋' },
    { loudong_id: '6068', loudong_name: '18栋' },
    { loudong_id: '6267', loudong_name: '19栋' },
    { loudong_id: '6454', loudong_name: '20栋' },
    { loudong_id: '6899', loudong_name: '21栋' },
    { loudong_id: 'B1', loudong_name: '1号楼' },
    { loudong_id: 'B2', loudong_name: '2号楼' },
    { loudong_id: 'B3', loudong_name: '3号楼' },
    { loudong_id: 'B4', loudong_name: '4号楼' },
    { loudong_id: 'B5', loudong_name: '5号楼' },
    { loudong_id: 'B6', loudong_name: '6号楼' },
    { loudong_id: 'B7', loudong_name: '7号楼' },
    { loudong_id: 'B9', loudong_name: '9号楼' },
    { loudong_id: 'B10', loudong_name: '10号楼' },
    { loudong_id: 'B11', loudong_name: '11号楼' },
    { loudong_id: 'B12', loudong_name: '12号楼' },
    { loudong_id: 'B16', loudong_name: '16号楼' },
    { loudong_id: 'B19', loudong_name: '14号楼' },
    { loudong_id: 'B20', loudong_name: '致远楼一单元' },
    { loudong_id: 'B21', loudong_name: '致远楼二单元' },
    { loudong_id: 'B22', loudong_name: '德馨楼一单元' },
    { loudong_id: 'B23', loudong_name: '德馨楼二单元' },
    { loudong_id: 'B24', loudong_name: '博雅楼一单元' },
    { loudong_id: 'B25', loudong_name: '博雅楼二单元' }
];

async function getBuildings(accessToken, areaId = '') {
    try {
        if (areaId === 'glxq') {
            return GUILIN_BUILDINGS;
        }
        
        if (areaId === 'nnxq') {
            const roomUrl = getRoomUrl(areaId);
            const headers = {
                ...DEFAULT_HEADERS,
                'Authorization': `bearer ${accessToken}`,
                'Content-Type': 'multipart/form-data'
            };
            
            const formData = new FormData();
            formData.append('areaId', areaId);
            
            try {
                const response = await axios.post(`${roomUrl}/v1/cgElec/loudong/query`, formData, {
                    headers
                });
                
                const apiBuildings = response.data?.data || response.data || [];
                const existingIds = new Set(apiBuildings.map(b => b.loudong_id));
                const newBuildings = NANNING_BUILDINGS.filter(b => !existingIds.has(b.loudong_id));
                
                return [...apiBuildings, ...newBuildings];
            } catch (apiError) {
                console.error('南宁校区API获取楼栋失败:', apiError.message);
                return NANNING_BUILDINGS;
            }
        }
        
        const roomUrl = getRoomUrl(areaId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        if (areaId) {
            formData.append('areaId', areaId);
        }
        
        const response = await axios.post(`${roomUrl}/v1/cgElec/loudong/query`, formData, {
            headers
        });
        
        if (response.data) {
            return response.data.data || response.data || [];
        }
        return [];
    } catch (error) {
        console.error('获取宿舍楼失败:', error.message);
        if (areaId === 'nnxq') {
            return NANNING_BUILDINGS;
        }
        return [];
    }
}

async function getRooms(accessToken, buildingId, areaId = '', page = 1, size = 100) {
    try {
        const roomUrl = getRoomUrl(areaId, buildingId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('loudong_id', buildingId);
        formData.append('current', page);
        formData.append('size', size);
        
        const response = await axios.post(`${roomUrl}/v1/cgElec/room/query`, formData, {
            headers
        });
        
        if (response.data) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取宿舍房间失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getGuilinRooms(accessToken, buildingId, page = 1, size = 100) {
    const roomUrl = getRoomUrl('glxq', buildingId);
    const headers = {
        ...DEFAULT_HEADERS,
        'Authorization': `bearer ${accessToken}`,
        'Content-Type': 'multipart/form-data'
    };
    
    const tryParams = [
        { dormitoryBdId: buildingId },
        { loudong_id: buildingId },
        { buildingId: buildingId },
        { id: buildingId }
    ];
    
    for (const params of tryParams) {
        try {
            const formData = new FormData();
            for (const [key, value] of Object.entries(params)) {
                formData.append(key, value);
            }
            formData.append('current', page);
            formData.append('size', size);
            
            console.log(`尝试桂林房间查询 URL: ${roomUrl}/v1/cgElec/room/query, 参数: ${JSON.stringify(params)}`);
            
            const response = await axios.post(`${roomUrl}/v1/cgElec/room/query`, formData, {
                headers,
                timeout: 10000
            });
            
            console.log(`响应状态: ${response.status}, 数据: ${JSON.stringify(response.data).substring(0, 500)}`);
            
            if (response.data && response.data.data && response.data.data.length > 0) {
                console.log(`成功获取 ${response.data.data.length} 个房间`);
                return {
                    data: response.data.data,
                    total: response.data.total || response.data.data.length,
                    pages: response.data.pages || 1,
                    current: response.data.current || page
                };
            }
        } catch (error) {
            console.error(`参数 ${JSON.stringify(params)} 失败:`, error.message);
            if (error.response) {
                console.error(`响应状态: ${error.response.status}, 数据: ${JSON.stringify(error.response.data)}`);
            }
        }
    }
    
    console.log(`桂林校区 ${buildingId} API查询失败，使用硬编码数据`);
    return { data: [], total: 0, pages: 0, current: page };
}

async function getNanningRooms(accessToken, buildingId, page = 1, size = 100) {
    const roomUrl = getRoomUrl('nnxq', buildingId);
    const headers = {
        ...DEFAULT_HEADERS,
        'Authorization': `bearer ${accessToken}`,
        'Content-Type': 'multipart/form-data'
    };
    
    const tryParams = [
        { dormitoryBdId: buildingId },
        { loudong_id: buildingId },
        { buildingId: buildingId }
    ];
    
    for (const params of tryParams) {
        try {
            const formData = new FormData();
            for (const [key, value] of Object.entries(params)) {
                formData.append(key, value);
            }
            formData.append('current', page);
            formData.append('size', size);
            
            console.log(`尝试南宁房间查询 URL: ${roomUrl}/v1/cgElec/room/query, 参数: ${JSON.stringify(params)}`);
            
            const response = await axios.post(`${roomUrl}/v1/cgElec/room/query`, formData, {
                headers,
                timeout: 10000
            });
            
            console.log(`响应状态: ${response.status}, 数据: ${JSON.stringify(response.data).substring(0, 500)}`);
            
            if (response.data && response.data.data && response.data.data.length > 0) {
                console.log(`成功获取 ${response.data.data.length} 个房间`);
                return {
                    data: response.data.data,
                    total: response.data.total || response.data.data.length,
                    pages: response.data.pages || 1,
                    current: response.data.current || page
                };
            }
        } catch (error) {
            console.error(`参数 ${JSON.stringify(params)} 失败:`, error.message);
            if (error.response) {
                console.error(`响应状态: ${error.response.status}, 数据: ${JSON.stringify(error.response.data)}`);
            }
        }
    }
    
    console.log(`南宁校区 ${buildingId} API查询失败，使用硬编码数据`);
    return { data: [], total: 0, pages: 0, current: page };
}

async function getRoomsByDormitoryBdId(accessToken, dormitoryBdId, areaId = '') {
    try {
        const elecUrl = getElecUrl(areaId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('dormitoryBdId', dormitoryBdId);
        
        const response = await axios.post(`${elecUrl}/v1/cgElec/room/query`, formData, {
            headers
        });
        
        console.log(`getRoomsByDormitoryBdId response for ${dormitoryBdId}:`, JSON.stringify(response.data));
        
        if (response.data && response.data.data) {
            return response.data.data || [];
        }
        return [];
    } catch (error) {
        console.error('getRoomsByDormitoryBdId失败:', error.message);
        return [];
    }
}

function generateGuilinRooms(prefix, buildingName, floors, roomsPerFloor) {
    const rooms = [];
    for (let floor = 1; floor <= floors; floor++) {
        for (let room = 1; room <= roomsPerFloor; room++) {
            const roomNum = floor * 100 + room;
            const roomId = `${prefix}${roomNum}`;
            const roomName = `${buildingName}${roomNum}`;
            rooms.push({ room_id: roomId, room_name: roomName });
        }
    }
    return rooms;
}

const GUILIN_BUILDING_ROOMS = {
    '4320': generateGuilinRooms('H432', '9栋', 6, 20),
    '4509': generateGuilinRooms('H450', '7栋', 6, 20),
    '4722': generateGuilinRooms('H472', '12栋', 6, 20),
    '4812': generateGuilinRooms('H481', '13栋', 10, 20),
    '6436': generateGuilinRooms('H643', '14A栋', 10, 20),
    '6819': generateGuilinRooms('H681', '14B栋', 10, 20),
    'B101': generateGuilinRooms('HB101', '10A-', 6, 20),
    'B102': generateGuilinRooms('HB102', '10B-', 6, 20),
    'B8': generateGuilinRooms('HB8', '8-', 6, 20)
};

const NANNING_BUILDING_ROOMS = {
    '4320': generateGuilinRooms('H', '15-1-', 6, 20),
    '4523': generateGuilinRooms('H', '15-2-', 6, 20),
    '4722': generateGuilinRooms('H', '13-1-', 6, 20),
    '5158': generateGuilinRooms('H', '13-2-', 6, 20),
    '5623': generateGuilinRooms('H', '17-', 6, 20),
    '6068': generateGuilinRooms('H', '18-', 6, 20),
    '6267': generateGuilinRooms('H', '19-', 6, 20),
    '6454': generateGuilinRooms('H', '20-', 6, 20),
    '6899': generateGuilinRooms('H', '21-', 6, 20),
    'B1': generateGuilinRooms('H', '1-', 6, 20),
    'B2': generateGuilinRooms('H', '2-', 6, 20),
    'B3': generateGuilinRooms('H', '3-', 6, 20),
    'B4': generateGuilinRooms('H', '4-', 6, 20),
    'B5': generateGuilinRooms('H', '5-', 6, 20),
    'B6': generateGuilinRooms('H', '6-', 6, 20),
    'B7': generateGuilinRooms('H', '7-', 6, 20),
    'B8': generateGuilinRooms('H', '8-', 6, 20),
    'B9': generateGuilinRooms('H', '9-', 6, 20),
    'B10': generateGuilinRooms('H', '10-', 6, 20),
    'B11': generateGuilinRooms('H', '11-', 6, 20),
    'B12': generateGuilinRooms('H', '12-', 6, 20),
    'B16': generateGuilinRooms('H', '16-', 6, 20),
    'B19': generateGuilinRooms('H14', '14-', 10, 20),
    'B20': generateGuilinRooms('H', '致远楼一单元-', 6, 20),
    'B21': generateGuilinRooms('H', '致远楼二单元-', 6, 20),
    'B22': generateGuilinRooms('H', '德馨楼一单元-', 6, 20),
    'B23': generateGuilinRooms('H', '德馨楼二单元-', 6, 20),
    'B24': generateGuilinRooms('H', '博雅楼一单元-', 6, 20),
    'B25': generateGuilinRooms('H', '博雅楼二单元-', 6, 20)
};

async function getAllRoomsByBuilding(accessToken, buildingId, areaId = '') {
    if (areaId === 'glxq') {
        const allRooms = [];
        let page = 1;
        const size = 100;
        
        while (true) {
            const result = await getGuilinRooms(accessToken, buildingId, page, size);
            if (result.data && result.data.length > 0) {
                allRooms.push(...result.data);
            }
            if (page >= result.pages || result.data.length === 0) {
                break;
            }
            page++;
        }
        
        if (allRooms.length > 0) {
            console.log(`桂林校区 ${buildingId} 从API获取到 ${allRooms.length} 个房间`);
            return allRooms;
        }
        
        if (GUILIN_BUILDING_ROOMS[buildingId]) {
            console.log(`桂林校区 ${buildingId} 使用硬编码数据`);
            return GUILIN_BUILDING_ROOMS[buildingId];
        }
        
        return [];
    }
    
    if (areaId === 'nnxq') {
        const allRooms = [];
        let page = 1;
        const size = 100;
        
        while (true) {
            const result = await getNanningRooms(accessToken, buildingId, page, size);
            if (result.data && result.data.length > 0) {
                allRooms.push(...result.data);
            }
            if (page >= result.pages || result.data.length === 0) {
                break;
            }
            page++;
        }
        
        if (allRooms.length > 0) {
            console.log(`南宁校区 ${buildingId} 从API获取到 ${allRooms.length} 个房间`);
            return allRooms;
        }
        
        if (NANNING_BUILDING_ROOMS[buildingId]) {
            console.log(`南宁校区 ${buildingId} 使用硬编码数据`);
            return NANNING_BUILDING_ROOMS[buildingId];
        }
        
        return [];
    }
    
    const allRooms = [];
    let page = 1;
    const size = 100;
    
    while (true) {
        const result = await getRooms(accessToken, buildingId, areaId, page, size);
        if (result.data && result.data.length > 0) {
            allRooms.push(...result.data);
        }
        if (page >= result.pages || result.data.length === 0) {
            break;
        }
        page++;
    }
    
    return allRooms;
}

const BUILDINGS = {
    '15-1栋': '4320',
    '15-2栋': '4523',
    '13-1栋': '4722',
    '13-2栋': '5158',
    '17栋': '5623',
    '18栋': '6068',
    '19栋': '6267',
    '20栋': '6454',
    '21栋': '6899'
};

async function getAllBuildingsRooms(accessToken, areaId = 'nnxq') {
    const result = {};
    
    for (const [name, id] of Object.entries(BUILDINGS)) {
        console.log(`获取 ${name} 房间列表...`);
        const rooms = await getAllRoomsByBuilding(accessToken, id, areaId);
        result[name] = {
            buildingId: id,
            rooms: rooms
        };
    }
    
    return result;
}

async function getElectricity(accessToken, roomId, areaId = '') {
    try {
        const elecUrl = getElecUrlForRoom(areaId, roomId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('room_id', roomId);
        
        const response = await axios.post(`${elecUrl}/v1/cgElec/elec/query`, formData, {
            headers
        });
        
        if (response.data) {
            return response.data.data || response.data || null;
        }
        return null;
    } catch (error) {
        console.error('获取电费余额失败:', error.message);
        return null;
    }
}

module.exports = {
    login,
    refreshToken,
    getUserInfo,
    getBalance,
    getTransactions,
    getConsumptionRecords,
    getRechargeRecords,
    getBuildings,
    getRooms,
    getAllRoomsByBuilding,
    getAllBuildingsRooms,
    BUILDINGS,
    getElectricity,
    createInstance,
    formatCookies
};
